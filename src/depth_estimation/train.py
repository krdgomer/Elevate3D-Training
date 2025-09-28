# train.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

from src.depth_estimation.config import cfg
from src.depth_estimation.dataset import SatelliteDepthDataset
from src.depth_estimation.model import setup_model, setup_optimizer

def extract_predictions_dpt(outputs):
    """Correct way to extract predictions from DPT model"""
    # DPTForDepthEstimation returns DPTDepthEstimationOutput object
    # The predicted depth map is in outputs.predicted_depth
    if hasattr(outputs, 'predicted_depth'):
        return outputs.predicted_depth
    elif isinstance(outputs, dict) and 'predicted_depth' in outputs:
        return outputs['predicted_depth']
    else:
        # Fallback: try to access the raw output
        return outputs

def save_checkpoint(state, filename="checkpoint.pth"):
    """Save training checkpoint."""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(cfg.OUTPUT_DIR, filename)
    torch.save(state, path)

def validate_model_before_training(model, train_loader, device):
    """Validate that model can learn before full training"""
    print("🧪 Running pre-training validation...")
    
    model.train()
    criterion = torch.nn.MSELoss()
    
    # Get one batch
    rgb_batch, dsm_batch = next(iter(train_loader))
    rgb_batch = rgb_batch.to(device)
    dsm_batch = dsm_batch.to(device)
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(pixel_values=rgb_batch)
        predictions = extract_predictions_dpt(outputs)
        
        if len(predictions.shape) == 3:
            predictions = predictions.unsqueeze(1)
        
        print(f"Pre-train predictions range: [{predictions.min().item():.6f}, {predictions.max().item():.6f}]")
        
        if torch.all(predictions == 0):
            print("❌ CRITICAL: Model produces all zeros! Check architecture.")
            return False
        else:
            print("✅ Model produces non-zero outputs - ready for training")
            return True
        
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (rgb_batch, dsm_batch) in enumerate(pbar):
        rgb_batch = rgb_batch.to(device)
        dsm_batch = dsm_batch.to(device)
        
        # Debug first batch
        if batch_idx == 0:
            print(f"Input RGB batch shape: {rgb_batch.shape}")
            print(f"Target DSM batch shape: {dsm_batch.shape}")
        
        optimizer.zero_grad()
        
        # Forward pass - FIXED: Use correct output extraction
        outputs = model(pixel_values=rgb_batch)
        predictions = extract_predictions_dpt(outputs)
        
        if batch_idx == 0:
            print(f"Raw predictions shape: {predictions.shape}")
            print(f"Raw predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
        
        # Ensure predictions have the right shape [B, H, W] -> [B, 1, H, W]
        if len(predictions.shape) == 3:
            predictions = predictions.unsqueeze(1)
        
        # Resize predictions to match target size if needed
        if predictions.shape[-2:] != dsm_batch.shape[-2:]:
            predictions = F.interpolate(
                predictions, 
                size=dsm_batch.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        if batch_idx == 0:
            print(f"Final predictions shape: {predictions.shape}")
            print(f"Final predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
            print("-" * 50)
        
        # Calculate loss
        loss = criterion(predictions, dsm_batch)
        
        # Backward pass
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
        for batch_idx, (rgb_batch, dsm_batch) in enumerate(pbar):
            rgb_batch = rgb_batch.to(device)
            dsm_batch = dsm_batch.to(device)
            
            # Debug first batch
            if batch_idx == 0:
                print(f"Val Input RGB shape: {rgb_batch.shape}")
                print(f"Val Target DSM shape: {dsm_batch.shape}")
            
            # Forward pass - FIXED: Use correct output extraction
            outputs = model(pixel_values=rgb_batch)
            predictions = extract_predictions_dpt(outputs)
            
            if batch_idx == 0:
                print(f"Val Raw predictions shape: {predictions.shape}")
                print(f"Val Raw predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
            
            # Ensure predictions have the right shape [B, H, W] -> [B, 1, H, W]
            if len(predictions.shape) == 3:
                predictions = predictions.unsqueeze(1)
            
            # Resize predictions to match target size if needed
            if predictions.shape[-2:] != dsm_batch.shape[-2:]:
                predictions = F.interpolate(
                    predictions, 
                    size=dsm_batch.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            if batch_idx == 0:
                print(f"Val Final predictions shape: {predictions.shape}")
                print(f"Val Final predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
                print("-" * 50)
            
            loss = criterion(predictions, dsm_batch)
            epoch_loss += loss.item() * rgb_batch.size(0)
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
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
    train_dataset = SatelliteDepthDataset(
        rgb_dir=cfg.RGB_DIR, 
        dsm_dir=cfg.DSM_DIR,
        batch_size_for_stats=50,
        resize_to_model_input=True  # This handles 512x512 -> 384x384 resize
    )
    
    val_dataset = SatelliteDepthDataset(
        rgb_dir=cfg.VAL_RGB_DIR, 
        dsm_dir=cfg.VAL_DSM_DIR,
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = torch.nn.HuberLoss(delta=0.5)
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Training on device: {cfg.DEVICE}")
    
    best_val_loss = float('inf')
    if not validate_model_before_training(model, train_loader, cfg.DEVICE):
        print("Stopping training due to model issues")
        return
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