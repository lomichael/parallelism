from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_dataloader(batch_size, split="train"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(tokenized_datasets, batch_size=batch_size)
    return dataloader

