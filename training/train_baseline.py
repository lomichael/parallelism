import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from training.utils import train_one_epoch
from data.datasets import get_dataloader
import logging

def main():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    dataloader = get_dataloader(tokenizer, batch_size=32, dataset_name="wikitext-2", split="train")

    optimizer = Adam(model.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss()

    for epoch in range(3):
        loss, epoch_time = train_one_epoch(model, dataloader, optimizer, criterion, device, description=f"Training Baseline Epoch {epoch+1}")
        logging.info(f"Epoch {epoch+1} - Baseline Training Loss: {loss}, Time: {epoch_time} seconds")

if __name__ == "__main__":
    main()

