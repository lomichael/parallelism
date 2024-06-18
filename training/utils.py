import torch
from tqdm import tqdm
import logging

def train_one_epoch(model, dataloader, optimizer, criterion, device, description="Training"):
    model.train()
    total_loss = 0.0
    epoch_start_time = torch.cuda.Event(enable_timing=True)
    epoch_end_time = torch.cuda.Event(enable_timing=True)
    epoch_start_time.record()

    for batch in tqdm(dataloader, desc=description):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        
        logging.info(f"Batch input IDs device: {input_ids.device}")
        logging.info(f"Batch attention mask device: {attention_mask.device}")

        logits = model(input_ids, attention_mask=attention_mask)

        logging.info(f"Logits device: {logits.device}")
        logging.info(f"Input IDs device for loss: {input_ids.device}")

        input_ids = input_ids.to(logits.device)
        
        loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    epoch_end_time.record()
    torch.cuda.synchronize()
    epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0  # Convert to seconds

    return total_loss / len(dataloader), epoch_time

