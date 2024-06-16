import torch
import torch.optim as optim
import torch.nn as nn
from data.dataset import get_dataloader
from model.model import SimpleGPT2
from training.utils import train_one_epoch
from parallelism.model_parallel import ModelParallel
from parallelism.pipeline_parallel import PipelineParallel
from parallelism.data_parallel import DataParallel
import logging
import time

# Setup logging
logging.basicConfig(filename='combined_parallel_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	model = SimpleGPT2().to(device)
	model = ModelParallel(model)
	model = PipelineParallel(model)
	model = DataParallel(model)  # Wrap model with combined parallelism
	dataloader = get_dataloader(batch_size=32, split='train')
	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	criterion = nn.CrossEntropyLoss()

	epochs = 5
	total_training_time = 0
	for epoch in range(epochs):
		loss, epoch_time = train_one_epoch(model, dataloader, optimizer, criterion, device)
		total_training_time += epoch_time
		logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Epoch Time: {epoch_time:.2f}s')
	
	logging.info(f'Total Training Time: {total_training_time:.2f}s')

if __name__ == "__main__":
	main()

