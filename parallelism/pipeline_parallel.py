import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.pipeline.sync import Pipe

class PipelineParallel(nn.Module):
	def __init__(self, model):
		super(PipelineParallel, self).__init__()
		self.model = model
		self.device0 = torch.device('cuda:0')
		self.device1 = torch.device('cuda:1')

		self.model.transformer.half1 = nn.Sequential(*self.model.transformer.h[:6]).to(self.device0)
		self.model.transformer.half2 = nn.Sequential(*self.model.transformer.h[6:]).to(self.device1)

		self.model = nn.Sequential(
			self.model.transformer.wte,
			self.model.transformer.drop,
			self.model.transformer.half1,
			self.model.transformer.half2,
			self.model.transformer.ln_f,
			self.model.lm_head,
		)

		self.model = Pipe(self.model, chunks=8, devices=[self.device0, self.device1])

	def forward(self, x):
		return self.model(x)

