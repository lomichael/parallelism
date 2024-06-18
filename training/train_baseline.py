import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from training.utils import train_one_epoch
from data.datasets import get_dataloader
import logging

logging.basicConfig(level=logging.INFO, filename="training.log", filemode="w")

def main():
    torch.cuda.empty_cache()
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 4  # Reduced batch size
    dataloader = get_dataloader(tokenizer, batch_size=batch_size, dataset_name="wikitext-2", split="train")

    optimizer = Adam(model.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss()
    accumulation_steps = 8  # Gradient accumulation steps

    for epoch in range(3):
        loss, epoch_time = train_one_epoch(model, dataloader, optimizer, criterion, device, description=f"Training Baseline Epoch {epoch+1}", accumulation_steps=accumulation_steps)
        logging.info(f"Epoch {epoch+1} - Baseline Training Loss: {loss}, Time: {epoch_time} seconds")

if __name__ == "__main__":
    main()

