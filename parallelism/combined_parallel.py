import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class CombinedParallel(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(CombinedParallel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

        # Split the model into parts for model parallelism
        self.embedding = self.model.transformer.wte
        self.embedding.to(self.devices[0])
        
        self.transformer_blocks_part1 = nn.ModuleList(self.model.transformer.h[:6]).to(self.devices[0])
        self.transformer_blocks_part2 = nn.ModuleList(self.model.transformer.h[6:]).to(self.devices[1])
        
        self.ln_f = self.model.transformer.ln_f
        self.ln_f.to(self.devices[0])
        
        self.lm_head = self.model.lm_head
        self.lm_head.to(self.devices[0])

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids).to(self.devices[0])
        
        for block in self.transformer_blocks_part1:
            x = block(x.to(self.devices[0]))[0]
        
        for block in self.transformer_blocks_part2:
            x = block(x.to(self.devices[1]))[0]
        
        x = self.ln_f(x.to(self.devices[0]))
        logits = self.lm_head(x.to(self.devices[0]))

        return logits

