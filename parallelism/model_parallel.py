import torch
import torch.nn as nn

class ModelParallel(nn.Module):
	def __init__(self, model):
		super(ModelParallel, self).__init__()
		self.model = model
		self.device0 = torch.device('cuda:0')
		self.device1 = torch.device('cuda:1')

		# Split the model into two parts
		self.model.transformer.half1 = nn.Sequential(*self.model.transformer.h[:6]).to(self.device0)
		self.model.transformer.half2 = nn.Sequential(*self.model.transformer.h[6:]).to(self.device1)

	def forward(self, x):
		x = x.to(self.device0)
		x = self.model.transformer.wte(x)
		x = self.model.transformer.drop(x)
		x = self.model.transformer.half1(x)
		x = x.to(self.device1)
		x = self.model.transformer.half2(x)
		x = self.model.transformer.ln_f(x)
		x = x.to(self.device0)
		x = self.model.lm_head(x)
		return x
