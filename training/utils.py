import torch
import time
from tqdm import tqdm
import logging

def train_one_epoch(model, dataloader, optimizer, criterion, device, description="Training", accumulation_steps=1):
    model.train()
    total_loss = 0
    start_time = time.time()

    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc=description)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        logging.info(f"Input IDs device: {input_ids.device}")
        logging.info(f"Attention mask device: {attention_mask.device}")
        logging.info(f"Logits device: {logits.device}")
        logging.info(f"Loss: {loss.item()}")

    end_time = time.time()
    epoch_time = end_time - start_time

    return total_loss / len(dataloader), epoch_time

