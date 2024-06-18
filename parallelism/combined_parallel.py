import torch.nn as nn
import torch
from torch.distributed.pipeline.sync import Pipe
from torch.nn.parallel import DataParallel as DP
from transformers import GPT2Config, GPT2LMHeadModel

class CombinedParallel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(CombinedParallel, self).__init__()

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

        self.pipeline_model = nn.Sequential(
            *self.transformer_blocks_part1,
            *self.transformer_blocks_part2,
            self.ln_f,
            self.lm_head
        )

        self.pipeline_model = Pipe(self.pipeline_model, chunks=8)
        self.pipeline_model = DP(self.pipeline_model, device_ids=[i for i in range(num_gpus)])

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.devices[0])
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.devices[0])

        position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)
        return self.pipeline_model(x, attention_mask=attention_mask)

