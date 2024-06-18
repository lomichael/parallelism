import os
import torch
import torch.distributed.rpc as rpc
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer
from training.utils import train_one_epoch
from parallelism.combined_parallel import CombinedParallel
from data.datasets import get_dataloader
import logging

logging.basicConfig(level=logging.INFO, filename="training_combined_parallel.log", filemode="w")

def main():
    torch.cuda.empty_cache()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    model_name = "gpt2"
    model = CombinedParallel(model_name=model_name)
    device = model.devices[0]

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 4  # Reduced batch size
    dataloader = get_dataloader(tokenizer, batch_size=batch_size, dataset_name="wikitext-2", split="train")

    optimizer = Adam(model.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss()
    accumulation_steps = 8  # Gradient accumulation steps

    rpc.init_rpc("worker", rank=0, world_size=1)

    try:
        for epoch in range(3):
            loss, epoch_time = train_one_epoch(model, dataloader, optimizer, criterion, device, description=f"Training Combined Parallel Epoch {epoch+1}", accumulation_steps=accumulation_steps)
            logging.info(f"Epoch {epoch+1} - Combined Parallel Training Loss: {loss}, Time: {epoch_time} seconds")
    finally:
        rpc.shutdown()

if __name__ == "__main__":
    main()

