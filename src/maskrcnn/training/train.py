import argparse
import torch
import os
from src.maskrcnn.model import get_model
import src.configs.maskrcnn_config as cfg
import numpy as np
from src.maskrcnn.training.dataset import MaskRCNNDataset, custom_collate
import torch.optim as optim
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description="Training Configuration")
parser.add_argument(
    "--train_images_path",
    type=str,
    required=True,
    help="Path to the training images directory",
)
parser.add_argument(
    "--train_masks_path",
    type=str,
    required=True,
    help="Path to the training masks directory",
)
parser.add_argument(
    "--val_images_path",
    type=str,
    required=True,
    help="Path to the validation images directory",
)
parser.add_argument(
    "--val_masks_path",
    type=str,
    required=True,
    help="Path to the validation masks directory",
)
parser.add_argument(
    "--save_path",
    type=str,
    required=True,
    help="Path to save the trained model",
)
args = parser.parse_args()

TRAIN_IMAGES_PATH = args.train_images_path
TRAIN_MASKS_PATH = args.train_masks_path
VAL_IMAGES_PATH = args.val_images_path
VAL_MASKS_PATH = args.val_masks_path
SAVE_PATH = args.save_path

def train():
    print("Loading training dataset...")
    train_images = sorted(os.listdir(TRAIN_IMAGES_PATH))
    train_masks = sorted(os.listdir(TRAIN_MASKS_PATH))

    print("Loading validation dataset...")
    val_images = sorted(os.listdir(VAL_IMAGES_PATH))
    val_masks = sorted(os.listdir(VAL_MASKS_PATH))

    print("Creating dataloaders...")
    train_dl = torch.utils.data.DataLoader(
        MaskRCNNDataset(train_images, train_masks, TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_dl = torch.utils.data.DataLoader(
        MaskRCNNDataset(val_images, val_masks, VAL_IMAGES_PATH, VAL_MASKS_PATH),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print("Initializing model...")
    model = get_model().to(cfg.DEVICE)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')  # Initialize with a very high value

    for epoch in range(cfg.NUM_EPOCHS):
        print(f"Starting epoch {epoch+1}/{cfg.NUM_EPOCHS}...")
        train_epoch_loss = 0
        val_epoch_loss = 0
        model.train()

        # Training loop with tqdm progress bar
        train_pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Training")
        for imgs, targets in train_pbar:
            imgs = [img.to(cfg.DEVICE) for img in imgs]
            targets = [{k: v.to(cfg.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)

            losses = sum(loss for loss in loss_dict.values())

            train_epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_pbar.set_postfix(loss=train_epoch_loss / len(train_dl))

        all_train_losses.append(train_epoch_loss / len(train_dl))
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Train Loss: {train_epoch_loss / len(train_dl):.4f}")

        # Validation loop
        print("Starting validation...")
        model.eval()
        with torch.no_grad():
            val_pbar = tqdm(val_dl, total=len(val_dl), desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Validation")
            for imgs, targets in val_pbar:
                imgs = [img.to(cfg.DEVICE) for img in imgs]
                targets = [{k: v.to(cfg.DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_epoch_loss += losses.item()
                val_pbar.set_postfix(loss=val_epoch_loss / len(val_dl))

        current_val_loss = val_epoch_loss / len(val_dl)
        all_val_losses.append(current_val_loss)
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Validation Loss: {current_val_loss:.4f}")

        # Save the model if it has the best validation loss so far
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            if cfg.SAVE_MODEL:
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                torch.save(model.state_dict(), os.path.join(SAVE_PATH, "maskrcnn_best_weights.pth"))
                print("Model saved successfully.")

    # Save the final model
    if cfg.SAVE_MODEL:
        print("Saving final model...")
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, "maskrcnn_final_weights.pth"))
        print("Final model saved successfully.")

if __name__ == "__main__":
    print("Starting training process...")
    train()
    print("Training complete.")