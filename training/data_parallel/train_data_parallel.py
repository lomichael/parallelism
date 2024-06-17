import sys
import os
import time
import logging
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.optim as optim
import torch.nn as nn
from data.dataset import get_dataloader
from model.simple_gpt2 import SimpleGPT2
from parallelism.data_parallel import DataParallel
from training.utils import train_one_epoch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'gpt2'

    model = SimpleGPT2(model_name).to(device)
    model = DataParallel(model)  # Wrap model with data parallelism

    dataloader = get_dataloader()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    logging.basicConfig(filename='data_parallel_training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    
    total_start_time = time.time()
    
    epochs = 3
    for epoch in tqdm(range(epochs), desc="Training Data Parallel"):
        loss, epoch_time = train_one_epoch(model, dataloader, optimizer, criterion, device)
        logging.info(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Time = {epoch_time:.2f}s")
    
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    logging.info(f"Total Training Time (Data Parallel): {total_training_time:.2f}s")

if __name__ == "__main__":
    main()

