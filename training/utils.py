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

	for input_ids, attention_mask in tqdm(dataloader, desc="Training", leave=False):
		input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

		optimizer.zero_grad()
		logits = model(input_ids, attention_mask=attention_mask)
		loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1)) # Flatten the outputs and labels
		loss.backward()
		optimizer.step()

		total_loss += loss.item()

	avg_loss = total_loss / len(dataloader)
	end_time = time.time()
	epoch_time = end_time - start_time # Time taken for the epoch
	logging.info(f'Epoch complete - Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')

	return avg_loss, epoch_time 
