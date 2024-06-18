import torch.nn as nn
import torch
from transformers import GPT2LMHeadModel, GPT2Config

class ModelParallel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(ModelParallel, self).__init__()

        num_gpus = torch.cuda.device_count()
        assert num_gpus >= 4, "This model requires at least 4 GPUs."

        self.devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]

        config = GPT2Config.from_pretrained(model_name)
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size).to(self.devices[0])
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size).to(self.devices[0])
        self.dropout = nn.Dropout(config.embd_pdrop).to(self.devices[0])

        self.transformer_blocks_part1 = nn.ModuleList(
            [GPT2LMHeadModel(config).transformer.h[i].to(self.devices[i % num_gpus]) for i in range(6)]
        )
        self.transformer_blocks_part2 = nn.ModuleList(
            [GPT2LMHeadModel(config).transformer.h[i].to(self.devices[(i + 6) % num_gpus]) for i in range(6, 12)]
        )

        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon).to(self.devices[-1])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(self.devices[-1])

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.devices[0])
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.devices[0])

        position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)

        for i, block in enumerate(self.transformer_blocks_part1):
            device = self.devices[i % len(self.devices)]
            x = x.to(device)
            x = block(x)[0]

        for i, block in enumerate(self.transformer_blocks_part2):
            device = self.devices[(i + 6) % len(self.devices)]
            x = x.to(device)
            x = block(x)[0]

        x = x.to(self.devices[-1])
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.devices[-1])

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

