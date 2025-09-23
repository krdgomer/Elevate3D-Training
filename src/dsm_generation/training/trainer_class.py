import torch
from src.utils.utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
from src.dsm_generation.dataset import MapDataset
from src.dsm_generation.generator import Generator
from src.dsm_generation.discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dsm_generation import config as cfg
import matplotlib.pyplot as plt
from src.dsm_generation.training.loss_function import ElevationLoss


class Pix2PixTrainer:
    def __init__(self, train_dir, val_dir):
        self._initialize_models()
        self._initialize_optimizers() 
        self._initialize_loss_functions()
        self._initialize_datasets(train_dir, val_dir)
        self._initialize_scalers()
        self.discriminator_losses = []
        self.generator_losses = []
    
    def _initialize_models(self):
        self.discriminator = Discriminator(in_channels=1).to(cfg.DEVICE) #in channels 1 because we are using grayscale images
        self.generator = Generator(in_channels=1, features=64).to(cfg.DEVICE)

    def _initialize_optimizers(self):
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(),lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))

    def _initialize_loss_functions(self):
        self.BCE = nn.BCEWithLogitsLoss()
        self.elevation_loss= ElevationLoss(
            base_weight=1.0,
            critical_range_weight=2.0,
            critical_range=(144, 200),
            perceptual_weight=0.1,
        ).to(cfg.DEVICE)

    def _initialize_datasets(self,train_dir, val_dir):
        self.train_dataset = MapDataset(root_dir=train_dir)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
        )
        self.validation_dataset = MapDataset(root_dir=val_dir)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=1, shuffle=False)

    def _initialize_scalers(self):
        self.generator_scaler = torch.amp.GradScaler(cfg.DEVICE)
        self.discriminator_scaler = torch.amp.GradScaler(cfg.DEVICE)

    def train(self):
        for epoch in range(cfg.NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}")
            try:
                avg_disc_loss, avg_gen_loss = self._train_one_epoch()
                print(f"Returned losses - Disc: {avg_disc_loss:.4f}, Gen: {avg_gen_loss:.4f}")
                self.discriminator_losses.append(avg_disc_loss)
                self.generator_losses.append(avg_gen_loss)
            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue
            if cfg.SAVE_MODEL and epoch % 5 == 0:
                save_checkpoint(self.generator, self.generator_optimizer, filename=cfg.CHECKPOINT_GEN)
                save_checkpoint(self.discriminator, self.discriminator_optimizer, filename=cfg.CHECKPOINT_DISC)

        self._plot_losses()

    def _train_one_epoch(self):
        loop = tqdm(self.train_loader, leave=True)
        total_disc_loss = 0
        total_gen_loss = 0

        # Add a check for empty loader
        if len(self.train_loader) == 0:
            return 0.0, 0.0  # Return default values if loader is empty

        for idx, (input_image, target_image) in enumerate(loop):
            try:
                input_image = input_image.to(cfg.DEVICE)
                target_image = target_image.to(cfg.DEVICE)

                D_loss,fake_image,D_real,D_fake = self._train_discriminator(input_image, target_image)
                total_disc_loss += D_loss

                # Train generator
                G_loss = self._train_generator(input_image, target_image,fake_image)
                total_gen_loss += G_loss

                # Accumulate losses
                total_disc_loss += D_loss.item()
                total_gen_loss += G_loss.item()

                if idx % 10 == 0:
                    loop.set_postfix(
                        D_real=torch.sigmoid(D_real).mean().item(),
                        D_fake=torch.sigmoid(D_fake).mean().item(),
                    )
            except Exception as e:
                print(f"Error during training iteration {idx}: {str(e)}")
                continue
            try:
                avg_disc_loss = total_disc_loss / len(self.train_loader)
                avg_gen_loss = total_gen_loss / len(self.train_loader)
            except ZeroDivisionError:
                print("Warning: Empty loader or division by zero")
                avg_disc_loss = total_disc_loss
                avg_gen_loss = total_gen_loss

            return avg_disc_loss, avg_gen_loss

    def _train_discriminator(self,input_image, target_image):   
        # Train Discriminator. Computes real (D_real) and fake (D_fake) losses using Binary Cross-Entropy (BCE).
        with torch.amp.autocast(cfg.DEVICE):
            fake_image = self.generator(input_image)
            D_real = self.discriminator(input_image, target_image)
            D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
            D_fake = self.discriminator(input_image, fake_image.detach())
            D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        # Updates the discriminator weights to better distinguish between real and fake samples.
        self.discriminator.zero_grad()
        self.discriminator_scaler.scale(D_loss).backward()
        self.discriminator_scaler.step(self.discriminator_optimizer)
        self.discriminator_scaler.update()
        return D_loss.item(),fake_image,D_real,D_fake

    def _train_generator(self,input_image, target_image,fake_image):
        with torch.amp.autocast(cfg.DEVICE):
            D_fake = self.discriminator(input_image, fake_image)
            G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
            L1 = self.elevation_loss(fake_image, target_image) * cfg.L1_LAMBDA
            G_loss = G_fake_loss + L1

        # Updates the generator weights to minimize the loss between the generated image and the target image.
        self.generator_optimizer.zero_grad()
        self.generator_scaler.scale(G_loss).backward()
        self.generator_scaler.step(self.generator_optimizer)
        self.generator_scaler.update()
        return G_loss.item()
    
    def _plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.discriminator_losses, label="Discriminator Loss")
        plt.plot(self.generator_losses, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss During Training")
        plt.savefig("training_loss_plot.png")  
        plt.show()