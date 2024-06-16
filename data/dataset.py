import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import GPT2Tokenizer

class WikiText2Dataset(Dataset):
	def __init__(self, split='train'): 
		self.dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split) 
		self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		self.tokenizer.pad_token = self.tokenizer.eos_token
		self.data = self.tokenizer(
			self.dataset['text'],
			return_tensors='pt',
			max_length=128,
			truncation=True,
			padding='max_length'
		)

	def __len__(self):
		return len(self.data['input_ids']) 

	def __getitem__(self, idx):
		input_ids = self.data['input_ids'][idx]
		attention_mask = self.data['attention_mask'][idx]
		return input_ids, attention_mask 

def get_dataloader(batch_size=32, split='train'):
	dataset = WikiText2Dataset(split=split)
	return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
	# Test the dataloader
	dataloader = get_dataloader()
	for input_ids, attention_mask in dataloader:
		print(input_ids.shape, attention_mask.shape)
		break
