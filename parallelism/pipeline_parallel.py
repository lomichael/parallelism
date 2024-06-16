import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe

class PipelineParallel(nn.Module):
    def __init__(self, model):
        super(PipelineParallel, self).__init__()
        self.device0 = torch.device('cuda:0')
        self.device1 = torch.device('cuda:1')

        # Split the model into pipeline stages
        self.embedding = nn.Sequential(
            model.transformer.wte,
            model.transformer.wpe,
            model.transformer.drop
        ).to(self.device0)

        self.transformer_part1 = nn.Sequential(*model.transformer.h[:6]).to(self.device0)
        self.transformer_part2 = nn.Sequential(*model.transformer.h[6:]).to(self.device1)
        self.ln_f = model.transformer.ln_f.to(self.device1)
        self.lm_head = model.lm_head.to(self.device1)

        # Create the pipeline model
        self.pipeline_model = nn.Sequential(
            self.embedding,
            self.transformer_part1,
            self.transformer_part2,
            self.ln_f,
            self.lm_head
        )

        self.pipeline_model = Pipe(self.pipeline_model, chunks=8, devices=[self.device0, self.device1])

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device0)
        return self.pipeline_model(input_ids, attention_mask=attention_mask)

