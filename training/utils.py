import torch
from tqdm import tqdm
import logging
import time

# Setup logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_one_epoch(model, dataloader, optimizer, criterion, device):
	model.train()
	total_loss = 0
	correct = 0
	total = 0
	start_time = time.time()

	for batch in tqdm(dataloader, desc="Training", leave=False):
		batch = batch.to(device)
		labels = torch.randint(0, 1000, (batch.size(0), batch.size(1))).to(device) # Adapt labels for GPT-2

		optimizer.zero_grad()
		outputs = model(batch)
		loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1)) # Flatten the outputs and labels
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		_, predicted = outputs.max(2)
		correct += predicted.eq(labels).sum().item()
		total += labels.numel()

	avg_loss = total_loss / len(dataloader)
	accuracy = correct / total
	end_time = time.time()
	epoch_time = end_time - start_time # Time taken for the epoch
	logging.info(f'Epoch complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.2f}s')

	return avg_loss, accuracy, epoch_time 
