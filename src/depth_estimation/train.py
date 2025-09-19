# train.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

from src.depth_estimation.config import cfg
from src.depth_estimation.dataset import SatelliteDepthDatasetWithCache
from src.depth_estimation.model import setup_model, setup_optimizer

def save_checkpoint(state, filename="checkpoint.pth"):
    """Save training checkpoint."""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(cfg.OUTPUT_DIR, filename)
    torch.save(state, path)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (rgb_batch, dsm_batch) in enumerate(pbar):
        rgb_batch = rgb_batch.to(device)
        dsm_batch = dsm_batch.to(device)
        
        # Debug first batch shapes
        if batch_idx == 0:
            print(f"Input RGB batch shape: {rgb_batch.shape}")  # Should be [B, 3, 384, 384]
            print(f"Target DSM batch shape: {dsm_batch.shape}")  # Should be [B, 1, 384, 384]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(pixel_values=rgb_batch)
        
        # Extract predictions from model output
        if hasattr(outputs, 'predicted_depth'):
            predictions = outputs.predicted_depth
        elif hasattr(outputs, 'prediction'):
            predictions = outputs.prediction
        elif hasattr(outputs, 'last_hidden_state'):
            predictions = outputs.last_hidden_state
        elif isinstance(outputs, dict):
            # Try different possible keys
            predictions = outputs.get('predicted_depth', 
                         outputs.get('prediction', 
                         outputs.get('logits', 
                         outputs.get('last_hidden_state'))))
        else:
            predictions = outputs
        
        if batch_idx == 0:
            print(f"Raw predictions shape: {predictions.shape}")
        
        # Ensure predictions have the right shape [B, 1, H, W]
        if len(predictions.shape) == 3:
            # [B, H, W] -> [B, 1, H, W]
            predictions = predictions.unsqueeze(1)
        elif len(predictions.shape) == 4 and predictions.shape[1] != 1:
            # If it has multiple channels, average them or take the first
            if predictions.shape[1] > 1:
                predictions = predictions.mean(dim=1, keepdim=True)
        
        # Resize predictions to match target size if needed
        if predictions.shape[-2:] != dsm_batch.shape[-2:]:
            if batch_idx == 0:
                print(f"Resizing predictions from {predictions.shape[-2:]} to {dsm_batch.shape[-2:]}")
            predictions = F.interpolate(
                predictions, 
                size=dsm_batch.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        if batch_idx == 0:
            print(f"Final predictions shape: {predictions.shape}")
            print(f"Final targets shape: {dsm_batch.shape}")
            print("-" * 50)
        
        # Calculate loss
        loss = criterion(predictions, dsm_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
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
            
            # Forward pass
            outputs = model(pixel_values=rgb_batch)
            
            # Extract predictions
            if hasattr(outputs, 'predicted_depth'):
                predictions = outputs.predicted_depth
            elif hasattr(outputs, 'prediction'):
                predictions = outputs.prediction
            elif hasattr(outputs, 'last_hidden_state'):
                predictions = outputs.last_hidden_state
            elif isinstance(outputs, dict):
                predictions = outputs.get('predicted_depth', 
                             outputs.get('prediction', 
                             outputs.get('logits', 
                             outputs.get('last_hidden_state'))))
            else:
                predictions = outputs
            
            # Ensure correct shape
            if len(predictions.shape) == 3:
                predictions = predictions.unsqueeze(1)
            elif len(predictions.shape) == 4 and predictions.shape[1] != 1:
                if predictions.shape[1] > 1:
                    predictions = predictions.mean(dim=1, keepdim=True)
            
            # Resize if needed
            if predictions.shape[-2:] != dsm_batch.shape[-2:]:
                predictions = F.interpolate(
                    predictions, 
                    size=dsm_batch.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            loss = criterion(predictions, dsm_batch)
            epoch_loss += loss.item() * rgb_batch.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / len(val_loader.dataset)

def debug_data_loading(dataset, device, num_samples=3):
    """Debug function to verify data loading works correctly"""
    print("=" * 60)
    print("DEBUGGING DATA LOADING FOR 8-BIT PNG FILES")
    print("=" * 60)
    
    for i in range(min(num_samples, len(dataset))):
        rgb, dsm = dataset[i]
        print(f"Sample {i}:")
        print(f"  RGB - Shape: {rgb.shape}, dtype: {rgb.dtype}")
        print(f"  RGB - Range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"  DSM - Shape: {dsm.shape}, dtype: {dsm.dtype}")
        print(f"  DSM - Range: [{dsm.min():.3f}, {dsm.max():.3f}]")
        print()
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    rgb_batch, dsm_batch = next(iter(loader))
    print("DataLoader batch test:")
    print(f"  RGB batch shape: {rgb_batch.shape}")  # Should be [2, 3, 384, 384]
    print(f"  DSM batch shape: {dsm_batch.shape}")  # Should be [2, 1, 384, 384]
    print("=" * 60)

def main():
    print("Setting up datasets for 8-bit PNG files (512x512 -> 384x384)...")
    
    # Setup datasets with caching to avoid recomputing stats
    train_dataset = SatelliteDepthDatasetWithCache(
        rgb_dir=cfg.RGB_DIR, 
        dsm_dir=cfg.DSM_DIR,
        stats_cache_file=os.path.join(cfg.OUTPUT_DIR, "train_dsm_stats.npz"),
        batch_size_for_stats=50,
        resize_to_model_input=True  # This handles 512x512 -> 384x384 resize
    )
    
    val_dataset = SatelliteDepthDatasetWithCache(
        rgb_dir=cfg.VAL_RGB_DIR, 
        dsm_dir=cfg.VAL_DSM_DIR,
        stats_cache_file=os.path.join(cfg.OUTPUT_DIR, "val_dsm_stats.npz"),
        batch_size_for_stats=50,
        resize_to_model_input=True
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Debug data loading
    print("\nDebugging training dataset...")
    debug_data_loading(train_dataset, cfg.DEVICE)
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,  # Adjust based on your system
        pin_memory=True if cfg.DEVICE.startswith('cuda') else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if cfg.DEVICE.startswith('cuda') else False
    )
    
    # Setup model and training components
    model = setup_model()
    optimizer = setup_optimizer(model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = torch.nn.HuberLoss(delta=0.5)
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Training on device: {cfg.DEVICE}")
    
    best_val_loss = float('inf')
    
    # Training loop
    print(f"\nStarting training for {cfg.NUM_EPOCHS} epochs...")
    for epoch in range(cfg.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{cfg.NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, cfg.DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'dsm_stats': {
                    'train_mean': train_dataset.dsm_mean,
                    'train_std': train_dataset.dsm_std,
                    'val_mean': val_dataset.dsm_mean,
                    'val_std': val_dataset.dsm_std
                }
            }, filename=cfg.BEST_MODEL_NAME)
            print(f'✓ Saved new best model with val loss: {val_loss:.6f}')
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, filename=f"checkpoint_epoch_{epoch+1}.pth")
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()