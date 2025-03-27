import torch
import argparse
from training.trainer_class import Pix2PixTrainer

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    # Argument parser
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to the training dataset directory",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Path to the validation dataset directory",
    )
    args = parser.parse_args()

    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir

    pix2pix_trainer = Pix2PixTrainer(TRAIN_DIR, VAL_DIR)
    pix2pix_trainer.train()
