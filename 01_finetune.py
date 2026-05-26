import argparse
import os
import random
import sys
import shutil
import time
from contextlib import nullcontext
from functools import partial
from tqdm import tqdm

# Before importing transformers !!
# os.environ["HF_HOME"] = ""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase)
from peft import LoraConfig, TaskType, get_peft_model


seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")


#===== Config =============================================================

config = {
    'epoch': 8,
    'batch': 32,
    'micro-batch' : {
        'microsoft/mdeberta-v3-base': 32,
        'FacebookAI/xlm-roberta-base': 32,
        'meta-llama/Llama-3.2-3B': 8,
        'google/gemma-2-2b': 4,
        'google/gemma-2-9b': 1,
        'google/madlad400-8b-lm': 4,
    },
    'lr': 3e-4,
    'weight-decay': 0.01,
    '16-mixed': True,
    'use-peft': True,
    'max-context-size': 512
}

all_trainig_languages =["de", "pl", "hr", "hu", "cs"]
all_eval_languages = ["de", "pl", "hr", "hu", "cs", "sl", "sk"]
tokenizer_pad_token_id = None

#===== Filtering training dataframe =============================================================

def estimate_low_resource_sample_size(df_all: pd.DataFrame):
    df_train = df_all[df_all['split'] == 'train']
    min_samples = float('inf')
    for lang in all_trainig_languages:
        df_lang = df_train[df_train['language'] == lang]
        df_lang_news = df_lang[df_lang['domain'] == 'news']
        df_lang_social = df_lang[df_lang['domain'] == 'social_media']
        news_human_samples = (df_lang_news['label'] == 0).sum()
        social_human_samples = (df_lang_social['label'] == 0).sum()
        min_samples = min(min_samples, news_human_samples, social_human_samples)
    return min_samples
    
def filter_train_dataset(df_all: pd.DataFrame, languages: list[str], min_samples: int):
    to_sample = min_samples // len(languages)
    df_train = df_all[(df_all['split'] == 'train') & (df_all['language'].isin(languages))]
    dfs = []
    for lang in languages:
        df_lang = df_train[df_train['language'] == lang]
        df_news = df_lang[df_lang['domain'] == 'news']
        df_social = df_lang[df_lang['domain'] == 'social_media']
        df_news_human = df_news[df_news['label'] == 0].sample(to_sample)
        df_news_machine = df_news[df_news['label'] == 1].sample(to_sample)
        df_social_human = df_social[df_social['label'] == 0].sample(to_sample)
        df_social_machine = df_social[df_social['label'] == 1].sample(to_sample)
        dfs.extend([df_news_human, df_news_machine, df_social_human, df_social_machine])
    return pd.concat(dfs)
    
#===== Dataloader =============================================================

class DataframeWrapper(Dataset):
    def __init__(self, data: pd.DataFrame):
        self._data = data
        
    def __getitem__(self, index: int):
        return self._data.iloc[index]

    def __len__(self):
        return len(self._data)

def collate_fn(tokenizer: PreTrainedTokenizerBase, batch: list[pd.Series]) -> dict[str, torch.Tensor]:
    inpts = tokenizer([x.text for x in batch], padding="longest", truncation=True, max_length=config['max-context-size'], return_tensors="pt")
    outputs = torch.tensor([x.label for x in batch])
    return {"input_ids": inpts['input_ids'], "attention_mask": inpts['attention_mask'], "labels": outputs }
    
def create_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(args.model)    
    if (tokenizer.pad_token is None) and (tokenizer.eos_token is not None):
        tokenizer.pad_token = tokenizer.eos_token
    global tokenizer_pad_token_id
    tokenizer_pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    return tokenizer
    
def create_dataloader():

    # 1) Estimate sample size for lowest resource scenario
    min_samples = 986 # estimate_low_resource_sample_size(data)
    print(f"Low resource training samples: {4 * min_samples}")

    # 2) Filter dataframes to match lowest resouces scenario in size
    df_train = filter_train_dataset(df, args.languages, min_samples)
    df_test = df[test_set_mask]
    print("Train df", df_train)
    print("Test df", df_test)

    # 3) Construct dataloaders
    train_dataloader = DataLoader(
        DataframeWrapper(df_train),
        batch_size=config['micro-batch'][args.model],
        num_workers=8,
        prefetch_factor=2,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=partial(collate_fn, tokenizer)
    )
    test_dataloader = DataLoader(
        DataframeWrapper(df_test),
        batch_size=2 * config['micro-batch'][args.model], # Eval fits bigger batch into memory
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        collate_fn=partial(collate_fn, tokenizer)
    )
    print("-----------------------------------------------")
    print("Total training steps: ", int(round(config['epoch'] * len(train_dataloader.dataset) // config['batch'])))
    print(f"Epoch steps: {len(train_dataloader.dataset) // config['batch']}")  # type: ignore
    print("--")
    print(f"Train dataset size: {len(train_dataloader.dataset)}")  # type: ignore
    print("--")
    print(f"Test dataset size: {len(test_dataloader.dataset)}")  # type: ignore
    print("-----------------------------------------------")
    return train_dataloader, test_dataloader
    
#===== Model =============================================================

def load_model() -> torch.nn.Module:
    model_config = AutoConfig.from_pretrained(args.model)
    model_config.ignore_mismatched_sizes = True
    model_config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=model_config)
    if tokenizer_pad_token_id:
        model.config.pad_token_id = tokenizer_pad_token_id

    if config['use-peft']:
        lora_params = dict(task_type=TaskType.SEQ_CLS)
        if "gemma" in args.model:
            lora_params['target_modules'] = ["q_proj", "k_proj"] # "v_proj"
        peft_config = LoraConfig(**lora_params)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    if torch.cuda.device_count():
        model.cuda()
        
    return model # type: ignore
    
#===== Training =============================================================

def append_to_df(values: list[float], column_name: str) -> None:
    df[column_name] = -1.0
    df.loc[test_set_mask, column_name] = values

def move_batch_to_cuda(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if torch.cuda.device_count():
        for k in batch.keys():
            batch[k] = batch[k].cuda(non_blocking=True)
    return batch

def train_step_misc(output: dict[str, torch.Tensor], labels: torch.Tensor, train_acc: list[float], current_step: int):
    train_acc.append(100 * torch.mean(output["logits"].argmax(dim=-1) == labels, dtype=float).item())
    if len(train_acc) > 100:
        train_acc.pop(0)
    for avg in [1, 20, 50, 100]:
        log_writer.add_scalar(f"accuracy_{avg}", float(np.mean(train_acc[-avg:])), current_step)


def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight-decay'])
    train_acc, best_test_acc, start_time = [], 0, time.time()
    best_predictions = []
    steps_per_epoch = len(train_dataloder.dataset) // config['batch']
    accumulation_steps = config['batch'] // config['micro-batch'][args.model]
    assert accumulation_steps
    
    for epoch in range(config['epoch']):
        
        # Train
        model.train()
        for batch_id, batch in enumerate(train_dataloder):
            batch = move_batch_to_cuda(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16) if config['16-mixed'] else nullcontext():
                output = model.forward(**batch)
            output["loss"] /= accumulation_steps 
            output["loss"].backward()
            if (batch_id + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                train_step_misc(output, batch["labels"], train_acc, steps_per_epoch * epoch + (batch_id // accumulation_steps))
                now = time.time()
                print(f"Epoch: {epoch} | Step: {batch_id // accumulation_steps}/{steps_per_epoch} | Time: {(now - start_time):.2f}s | Loss: {output['loss'].item():.3f} | Acc: {train_acc[-1]:.2f}")
                start_time = now
    
        # Eval
        eval_start = time.time()
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_dataloader), desc="Test split evaluation") as pbar:
                correct, total, test_loss, is_machine_prob = 0, 0, 0, []
                for batch_id, batch in enumerate(test_dataloader):
                    batch = move_batch_to_cuda(batch)
                    with torch.autocast("cuda", dtype=torch.bfloat16) if config['16-mixed'] else nullcontext():
                        outputs = model.forward(**batch)
                    _, predicted = outputs["logits"].max(1)
                    test_loss += outputs["loss"].item() * batch["labels"].size(0)
                    total += batch["labels"].size(0)
                    correct += int(predicted.eq(batch["labels"]).sum().item())
                    is_machine_prob.extend(F.softmax(outputs["logits"], dim=1)[:,-1].tolist())
                    pbar.update(1)
                test_loss /= len(test_dataloader.dataset)
                test_accuracy = 100 * correct / total 
                log_writer.add_scalar("test_loss", test_loss, epoch)
                log_writer.add_scalar("test_accuracy", test_accuracy, epoch)
                print(f"===Test: Loss {test_loss:.3f} | Acc {test_accuracy:.2f} | Time {(time.time() - eval_start):.2f}s")

            # Dont save the model, just save predictions on test split
            append_to_df(is_machine_prob, f"{args.model}_is_machine_prob_epoch_{epoch}")
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                best_predictions = is_machine_prob
                if config['use-peft']:
                    model.save_pretrained(os.path.join(version_dir, "model"))
                else:
                    torch.save(model.state_dict(), os.path.join(version_dir, "model.pt"))

    # Save the best predictions
    append_to_df(best_predictions, f"{args.model}_is_machine_prob_best")
                

    
    
### Supoprted models
# microsoft/mdeberta-v3-base
# meta-llama/Llama-3.2-3B
# FacebookAI/xlm-roberta-base
# google/gemma-2-9b or google/gemma-2-2b
# TODO: google/madlad400-8b-lm 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name on hugging face")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--languages", type=str, nargs="+", required=True, help="Set of languages to train on")
    parser.add_argument("--log_to_save", type=str)
    args = parser.parse_args()

    for lang in args.languages:
        if lang not in all_trainig_languages:
            raise ValueError(f"Language {lang} not in {all_trainig_languages}")
            
    # Prepare logs
    base_dir = os.path.join("lightning_logs", args.model.split('/')[-1], f"{args.model.split('/')[-1]}_{'_'.join(args.languages)}")
    os.makedirs(base_dir, exist_ok=True)
    version_dir = os.path.join(base_dir, f"version_{len(os.listdir(base_dir)) + 1}")
    log_writer = SummaryWriter(log_dir=version_dir) # type: ignore
    shutil.copy(sys.argv[0], os.path.join(version_dir, os.path.basename(sys.argv[0])))

    df = pd.read_csv(args.data, index_col=0)
    test_set_mask = (df['split'] == 'test') & (df['language'].isin(all_eval_languages)) 
    
    tokenizer = create_tokenizer()
    train_dataloder, test_dataloader = create_dataloader()
    model = load_model()
    # TODO: Compilation is not supported in devana execution node
    # model = torch.compile(model)
    
    train()
    # Save resutls
    df.to_csv(os.path.join(version_dir, "data.csv"), index=False)

    # Copy logs if provided
    if args.log_to_save and os.path.exists(args.log_to_save):
        shutil.copy(args.log_to_save, os.path.join(version_dir, "log.txt"))

    