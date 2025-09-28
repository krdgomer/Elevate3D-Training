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
    
    # More strategic freezing - unfreeze decoder and head
    # Freeze only the backbone (encoder) initially
    for name, param in model.named_parameters():
        if "backbone" in name and "layer" in name:
            # Freeze early layers of backbone
            layer_num = int(name.split("layer")[1].split(".")[0])
            if layer_num <= 2:  # Freeze first 2 layers of backbone
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            # Unfreeze neck, fusion, and head
            param.requires_grad = True
    
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