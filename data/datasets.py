import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import datasets

class WikiText2Dataset(Dataset):
    def __init__(self, tokenizer, split):
        self.dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }

def get_dataloader(tokenizer, batch_size, dataset_name="wikitext-2", split="train"):
    if dataset_name == "wikitext-2":
        dataset = WikiText2Dataset(tokenizer, split)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

