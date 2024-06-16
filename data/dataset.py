import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
	def __init__(self, size=1000, seq_len=128):
		self.size = size
		self.seq_len = seq_len
		self.data = torch.randint(0, 1000, (size, seq_len))

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		return self.data[idx]

def get_dataloader(batch_size=32):
	dataset = DummyDataset()
	return DataLoader(dataset, batch_size=batch_size, shuffle=True)
