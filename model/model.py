import torch.nn as nn
from transformers import GPT2LMHeadModel

class SimpleGPT2(nn.Module):
	def __init__(self, model_name='gpt2'):
		super(SimpleGPT2, self).__init__()
		self.transformer = GPT2LMHeadModel.from_pretrained(model_name)

	def forward(self, input_ids, attention_mask=None):
		outputs = self.transformer(input_ids, attention_mask=attention_mask).logits
		return outputs.logits
