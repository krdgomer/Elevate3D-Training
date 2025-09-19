# config.py
import torch

class Config:
    # Data
    RGB_DIR = "/content/drive/MyDrive/ProjeDosyalari/anh/train/rgb/"
    DSM_DIR = "/content/drive/MyDrive/ProjeDosyalari/anh/train/dsm/"
    VAL_RGB_DIR = "/content/drive/MyDrive/ProjeDosyalari/anh/val/rgb/"
    VAL_DSM_DIR = "/content/drive/MyDrive/ProjeDosyalari/anh/val/dsm/"
    
    # Model
    PRETRAINED_MODEL_NAME = "Intel/dpt-hybrid-midas"
    
    # Training
    BATCH_SIZE = 4
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output
    OUTPUT_DIR = "outputs/models"
    BEST_MODEL_NAME = "best_dpt_model.pth"
    
# Create a config object to import elsewhere
cfg = Config()