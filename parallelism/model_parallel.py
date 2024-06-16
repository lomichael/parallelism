import torch
import torch.nn as nn

class ModelParallel(nn.Module):
	def __init__(self, model):
		super(ModelParallel, self).__init__()
		self.model = model
		self.device0 = torch.device('cuda:0')
		self.device1 = torch.device('cuda:1')

		# Split the model into two parts
		self.embedding = nn.Sequential(
			self.model.transformer.wte,
			self.model.transformer.wpe,
			self.model.transformer.drop
		).to(self.device0)

		self.transformer_part1 = nn.Sequential(*self.model.transformer.h[:6]).to(self.device0)
		self.transformer_part2 = nn.Sequential(*self.model.transformer.h[6:]).to(self.device1)
		self.ln_f = self.model.transformer.ln_f.to(self.device1)
		self.lm_head = self.model.lm_head.to(self.device1)

	def forward(self, input_ids, attention_mask=None):
		input_ids = input_ids.to(self.device0)
		if attention_mask is not None:
			attention_mask = attention_mask.to(self.device0)
		
		x = self.embedding(input_ids)
		x = self.transformer_part1(x)
		x = x.to(self.device1)
		if attention_mask is not None:
			attention_mask = attention_mask.to(self.device1)
		x = self.transformer_part2(x, attention_mask=attention_mask)
		x = self.ln_f(x)
		logits = self.lm_head(x)
		return logits
