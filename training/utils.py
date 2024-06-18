import torch
from tqdm import tqdm
import logging
import time

# Setup logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for input_ids, attention_mask in tqdm(dataloader, desc="Training Epoch", leave=False):
        logging.info(f"Batch input IDs device: {input_ids.device}")
        input_ids = input_ids.to(device)
        logging.info(f"Batch input IDs device after moving to device: {input_ids.device}")

        if attention_mask is not None:
            logging.info(f"Batch attention mask device: {attention_mask.device}")
            attention_mask = attention_mask.to(device)
            logging.info(f"Batch attention mask device after moving to device: {attention_mask.device}")

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    end_time = time.time()
    epoch_time = end_time - start_time  # Time taken for the epoch
    logging.info(f'Epoch complete - Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')
    return avg_loss, epoch_time

