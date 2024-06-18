import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel
from data.datasets import get_dataloader
from training.utils import train_one_epoch
import logging
import os

logging.basicConfig(level=logging.INFO, filename="training_parallel.log", filemode="w")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    torch.cuda.empty_cache()
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    batch_size = 2  # Reduced batch size
    dataloader = get_dataloader(batch_size=batch_size, split="train")

    optimizer = Adam(model.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss()
    accumulation_steps = 8

    for epoch in range(3):
        loss, epoch_time = train_one_epoch(model, dataloader, optimizer, criterion, rank, description=f"Training Parallel Epoch {epoch+1}", accumulation_steps=accumulation_steps)
        if rank == 0:
            logging.info(f"Epoch {epoch+1} - Parallel Training Loss: {loss}, Time: {epoch_time} seconds")

    cleanup()

if __name__ == "__main__":
    world_size = 4
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)

