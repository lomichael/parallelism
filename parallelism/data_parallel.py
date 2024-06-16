import torch.nn as nn
from torch.nn.parallel import DataParallel as DP

class DataParallel(nn.Module):
	def __init__(self, model):
		super(DataParallel, self).__init__()
		self.model = DP(model)
	
	def forward(self, x):
		return self.model(x)

