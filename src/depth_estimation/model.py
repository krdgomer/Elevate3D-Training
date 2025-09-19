# model.py

import torch
import torch.nn as nn
from transformers import DPTForDepthEstimation
from torch.optim import AdamW
from src.depth_estimation.config import cfg
import os

def setup_model(device=cfg.DEVICE):
    """Initialize the DPT model with pre-trained weights."""
    model = DPTForDepthEstimation.from_pretrained(cfg.PRETRAINED_MODEL_NAME)
    
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the head for fine-tuning
    for param in model.head.parameters():
        param.requires_grad = True
        
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