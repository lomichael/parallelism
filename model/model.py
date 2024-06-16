import torch.nn as nn
from transformers import GPT2LMHeadModel

class SimpleGPT2(nn.Module):
	def __init__(self, model_name='gpt2'):
		super(SimpleGPT2, self).__init__()
		self.transformer = GPT2LMHeadModel.from_pretrained(model_name)

	def forward(self, x):
		return self.transformer(x).logits
