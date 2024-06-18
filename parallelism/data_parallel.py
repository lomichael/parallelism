import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
from transformers import GPT2LMHeadModel

class DataParallel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(DataParallel, self).__init__()
        
        num_gpus = torch.cuda.device_count()
        assert num_gpus >= 4, "This model requires at least 4 GPUs."

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model = DP(self.model, device_ids=[i for i in range(num_gpus)])

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)

