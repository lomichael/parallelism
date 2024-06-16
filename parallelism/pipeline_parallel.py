import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe
from transformers import GPT2Config, GPT2Block

class PipelineParallel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(PipelineParallel, self).__init__()
        self.device0 = torch.device('cuda:0')
        self.device1 = torch.device('cuda:1')

        # Load the model configuration
        config = GPT2Config.from_pretrained(model_name)

        # Manually create embedding layers and position embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size).to(self.device0)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size).to(self.device0)
        self.dropout = nn.Dropout(config.embd_pdrop).to(self.device0)

        # Manually create transformer blocks
        self.transformer_blocks_part1 = nn.ModuleList([GPT2Block(config).to(self.device0) for _ in range(6)])
        self.transformer_blocks_part2 = nn.ModuleList([GPT2Block(config).to(self.device1) for _ in range(6, 12)])

        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon).to(self.device1)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(self.device1)

        # Create the pipeline model
        self.pipeline_model = nn.Sequential(
            *self.transformer_blocks_part1,
            *self.transformer_blocks_part2,
            self.ln_f,
            self.lm_head
        )

        self.pipeline_model = Pipe(self.pipeline_model, chunks=8, devices=[self.device0, self.device1])

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device0)

        position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)
        return self.pipeline_model(x, attention_mask=attention_mask)

