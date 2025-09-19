# train.py

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

from config import cfg
from dataset import SatelliteDepthDataset
from model import setup_model, setup_optimizer
from utils import save_checkpoint  # We'll create this next

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for rgb_batch, dsm_batch in pbar:
        rgb_batch = rgb_batch.to(device)
        dsm_batch = dsm_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=rgb_batch)
        loss = criterion(outputs.predicted_depth, dsm_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item() * rgb_batch.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / len(train_loader.dataset)

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for rgb_batch, dsm_batch in pbar:
            rgb_batch = rgb_batch.to(device)
            dsm_batch = dsm_batch.to(device)
            
            outputs = model(pixel_values=rgb_batch)
            loss = criterion(outputs.predicted_depth, dsm_batch)
            
            epoch_loss += loss.item() * rgb_batch.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / len(val_loader.dataset)

def main():
    # Setup
    train_dataset = SatelliteDepthDataset(cfg.RGB_DIR, cfg.DSM_DIR)
    val_dataset = SatelliteDepthDataset(cfg.VAL_RGB_DIR, cfg.VAL_DSM_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    model = setup_model()
    optimizer = setup_optimizer(model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = torch.nn.HuberLoss(delta=0.5)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(cfg.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_loss = validate_epoch(model, val_loader, criterion, cfg.DEVICE)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1:02d}/{cfg.NUM_EPOCHS}')
        print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, filename=cfg.BEST_MODEL_NAME)
            print(f'Saved new best model with loss: {val_loss:.6f}')

if __name__ == "__main__":
    main()