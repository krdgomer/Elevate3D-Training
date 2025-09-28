# model.py

import torch
import torch.nn as nn
from transformers import DPTForDepthEstimation
from torch.optim import AdamW
from src.depth_estimation.config import cfg
import os

def setup_model(device=cfg.DEVICE):
    """Initialize the DPT model with proper fine-tuning setup."""
    model = DPTForDepthEstimation.from_pretrained(cfg.PRETRAINED_MODEL_NAME)
    
    # More strategic freezing - SIMPLIFIED APPROACH
    print("Setting up model with strategic freezing...")
    
    # Option 1: Freeze only the very early backbone layers, unfreeze everything else
    for name, param in model.named_parameters():
        # Freeze only the first few layers of the backbone
        if "backbone.embeddings" in name or "backbone.encoder.layer.0" in name:
            param.requires_grad = False
            print(f"❄️  Frozen: {name}")
        else:
            param.requires_grad = True
            print(f"🔥 Trainable: {name}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
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