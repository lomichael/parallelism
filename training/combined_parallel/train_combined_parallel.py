import sys
import os
import time
import logging
from tqdm import tqdm
from torch.distributed import rpc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.optim as optim
import torch.nn as nn
from data.dataset import get_dataloader
from parallelism.combined_parallel import CombinedParallel
from training.utils import train_one_epoch

def main():
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 4, "This training script requires at least 4 GPUs."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'gpt2'

    model = CombinedParallel(model_name)  # Wrap model with combined parallelism

    dataloader = get_dataloader()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    logging.basicConfig(filename='combined_parallel_training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    
    total_start_time = time.time()
    
    epochs = 3
    for epoch in tqdm(range(epochs), desc="Training Combined Parallel"):
        loss, epoch_time = train_one_epoch(model, dataloader, optimizer, criterion, device)
        logging.info(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Time = {epoch_time:.2f}s")
    
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    logging.info(f"Total Training Time: {total_training_time:.2f}s")

if __name__ == "__main__":
    rpc.init_rpc("worker", rank=0, world_size=1)
    main()
    rpc.shutdown()

