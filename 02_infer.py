import argparse
import random
import os
from contextlib import nullcontext
from tqdm import tqdm

# Before importing transformers !!
#os.environ["HF_HOME"] = ""

import numpy as np
import pandas as pd
import torch
from functools import partial

from torch.utils.data import Dataset, DataLoader

from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)

from peft import PeftModel


seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")

#===== Config =============================================================

config = {
    'batch' : {
        'microsoft/mdeberta-v3-base': 64,
        'FacebookAI/xlm-roberta-base': 64,
        'meta-llama/Llama-3.2-3B': 16,
        'google/gemma-2-2b': 16,
        'google/gemma-2-9b': 2,
        'google/madlad400-8b-lm': 8,
    },
    '16-mixed': True,
}

tokenizer_pad_token_id = None

#===== Dataloader =============================================================

class DataframeWrapper(Dataset):
    def __init__(self, data: pd.DataFrame):
        self._data = data
        
    def __getitem__(self, index: int) -> pd.Series:
        return self._data.iloc[index]

    def __len__(self) -> int:
        return len(self._data)

def collate_fn(tokenizer, batch: list[pd.Series]) -> dict[str, torch.Tensor]:
    inpts = tokenizer([sample.text for sample in batch], padding="longest", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inpts['input_ids']
    attention_mask = inpts['attention_mask']
    outputs = torch.tensor([sample.label for sample in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
    
def create_dataloader():
    tokenizer = AutoTokenizer.from_pretrained(args.model)    
    if (tokenizer.pad_token is None) and (tokenizer.eos_token is not None):
        tokenizer.pad_token = tokenizer.eos_token
    global tokenizer_pad_token_id
    tokenizer_pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
        
    df_test = df[test_set_mask]
    baseline_acc = df_test['label'].value_counts().max() / df.shape[0]
    print(f"Majority class: {100*baseline_acc:.2f}%")
    return DataLoader(
        DataframeWrapper(df_test),
        batch_size= config['batch'][args.model],
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer)
    )
    
#===== Model =============================================================

def init_model() -> torch.nn.Module:
    model_config = AutoConfig.from_pretrained(args.model)
    model_config.ignore_mismatched_sizes = True
    model_config.num_labels = 2
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model, config=model_config)
    if tokenizer_pad_token_id:
        base_model.config.pad_token_id = tokenizer_pad_token_id
    
    model = PeftModel.from_pretrained(base_model, args.path_to_weights)
    model.eval()
    if torch.cuda.device_count():
        model.cuda()
         
    return model # type: ignore
    
#===== Inference =============================================================

def move_batch_to_cuda(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if torch.cuda.device_count():
        for k in batch.keys():
            batch[k] = batch[k].cuda()
    return batch

def inference():
    softmax_fn = torch.nn.Softmax()
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Test split evaluation") as pbar:
            correct, total, test_loss, all_predictions = 0, 0, 0, []
            for _, batch in enumerate(dataloader):
                batch = move_batch_to_cuda(batch)
                with torch.autocast("cuda", dtype=torch.bfloat16) if config['16-mixed'] else nullcontext():
                    outputs = model.forward(**batch)
                _, predicted = outputs["logits"].max(1)
                all_predictions.extend(softmax_fn(outputs["logits"])[:,-1].tolist())
                test_loss += outputs["loss"].item() * batch["labels"].size(0)
                total += batch["labels"].size(0)
                correct += int(predicted.eq(batch["labels"]).sum().item())
                pbar.update(1)
            
        test_loss /= len(dataloader)
        test_accuracy = 100 * correct / total 
        print(f"===Test: Loss {test_loss:.3f} | Acc {test_accuracy:.2f}")

    return all_predictions

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name on hugging face")
    parser.add_argument("--path_to_weights", type=str, default=None, help="Path to finetuned weights")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--languages", type=str, nargs="+", help="Set of languages to evaluate on")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    test_set_mask = df['split'] == 'test'
    if args.languages:
        test_set_mask &= df['language'].isin(args.languages)

    dataloader = create_dataloader()
    model = init_model()
    predictions = inference()

    # Save predictions
    column_name = f"is_machine_prob_{args.model.split('/')[-1]}"
    df[column_name] = -1.0
    df.loc[test_set_mask, column_name] = predictions
    df.to_csv(args.data, index=False)

