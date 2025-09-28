# model.py

import torch
import torch.nn as nn
from transformers import DPTForDepthEstimation
from torch.optim import AdamW
from src.depth_estimation.config import cfg
import os

def setup_model(device=cfg.DEVICE):
    """Initialize the DPT model with minimal freezing."""
    model = DPTForDepthEstimation.from_pretrained(cfg.PRETRAINED_MODEL_NAME)
    
    print("Setting up model with minimal freezing...")
    
    # MINIMAL FREEZING: Start by unfreezing everything
    # This ensures the model can learn from your data
    for param in model.parameters():
        param.requires_grad = True
    
    # Optional: You can add gradual freezing later if needed
    # For now, let everything be trainable to ensure learning
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 All parameters trainable: {trainable_params:,} / {total_params:,} (100%)")
    
    model = model.to(device)
    return model

def setup_optimizer(model):
    """Create optimizer only for trainable parameters."""
    return AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

def load_saved_model(model, optimizer=None, path=None):
    """Load a saved model checkpoint."""
    if path is None:
        path = os.path.join(cfg.OUTPUT_DIR, cfg.BEST_MODEL_NAME)
        
    checkpoint = torch.load(path, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded model from {path}")
    return model, optimizer