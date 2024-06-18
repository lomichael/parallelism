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
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
            add_special_tokens=True  # Ensure special tokens are added
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # Ensure no sequence is empty
        if input_ids.size(0) == 0:
            input_ids = torch.tensor([self.tokenizer.eos_token_id] * 512)
            attention_mask = torch.tensor([1] * 512)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

def get_dataloader(tokenizer, batch_size, dataset_name="wikitext-2", split="train"):
    if dataset_name == "wikitext-2":
        dataset = WikiText2Dataset(tokenizer, split)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

