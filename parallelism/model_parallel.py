import torch.nn as nn
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import logging

# Configure logging to a file
logging.basicConfig(
    filename='model_parallel.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
            [GPT2LMHeadModel(config).transformer.h[i - 6].to(self.devices[i % num_gpus]) for i in range(6, 12)]
        )

        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon).to(self.devices[-1])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(self.devices[-1])

    def forward(self, input_ids, attention_mask=None):
        num_gpus = len(self.devices)
        
        logging.info(f"Input IDs device before embedding: {input_ids.device}")
        input_ids = input_ids.to(self.devices[0])
        logging.info(f"Input IDs device after moving to device[0]: {input_ids.device}")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.devices[0])
            logging.info(f"Attention mask device after moving to device[0]: {attention_mask.device}")

        position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        logging.info(f"Embedding output device: {x.device}")
        x = self.dropout(x)

        # Process part1 on respective devices
        for i, block in enumerate(self.transformer_blocks_part1):
            device = self.devices[i % num_gpus]
            x = x.to(device)
            logging.info(f"Block {i} part1 input device: {x.device}")
            x = block(x)[0]
            logging.info(f"Block {i} part1 output device: {x.device}")

            if i < len(self.transformer_blocks_part1) - 1:
                x = x.to(self.devices[(i + 1) % num_gpus])  # Ensure consistent device placement

        # Process part2 on respective devices
        for i, block in enumerate(self.transformer_blocks_part2):
            device = self.devices[i % num_gpus]
            x = x.to(device)
            logging.info(f"Block {i+6} part2 input device: {x.device}")
            x = block(x)[0]
            logging.info(f"Block {i+6} part2 output device: {x.device}")

            if i < len(self.transformer_blocks_part2) - 1:
                x = x.to(self.devices[(i + 1) % num_gpus])  # Ensure consistent device placement

        x = x.to(self.devices[-1])
        logging.info(f"Device after transformer blocks: {x.device}")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.devices[-1])
            logging.info(f"Attention mask device after moving to last device: {attention_mask.device}")

        x = self.ln_f(x)
        logging.info(f"LayerNorm output device: {x.device}")
        logits = self.lm_head(x)
        logging.info(f"Logits device: {logits.device}")
        return logits

