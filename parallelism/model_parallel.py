import torch.nn as nn
import torch
from transformers import GPT2LMHeadModel, GPT2Config

class ModelParallel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(ModelParallel, self).__init__()
        self.device0 = torch.device('cuda:0')
        self.device1 = torch.device('cuda:1')

        # Load the model
        model = GPT2LMHeadModel.from_pretrained(model_name)

        # Move the entire model to device0 first
        model.to(self.device0)

        # Manually split the embedding and transformer blocks for model parallelism
        self.embedding = nn.Embedding(model.config.vocab_size, model.config.hidden_size).to(self.device0)
        self.position_embedding = nn.Embedding(model.config.max_position_embeddings, model.config.hidden_size).to(self.device0)
        self.dropout = nn.Dropout(model.config.embd_pdrop).to(self.device0)

        # Splitting the transformer layers without directly accessing non-existent attributes
        self.transformer_blocks_part1 = nn.ModuleList([model.transformer.h[i].to(self.device0) for i in range(6)])
        self.transformer_blocks_part2 = nn.ModuleList([model.transformer.h[i].to(self.device1) for i in range(6, 12)])

        self.ln_f = nn.LayerNorm(model.config.hidden_size, eps=model.config.layer_norm_epsilon).to(self.device1)
        self.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False).to(self.device1)

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device0)

        position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)

        for block in self.transformer_blocks_part1:
            x = block(x)[0]  # GPT2 block returns a tuple, we need only the first item

        x = x.to(self.device1)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device1)

        for block in self.transformer_blocks_part2:
            x = block(x)[0]

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

