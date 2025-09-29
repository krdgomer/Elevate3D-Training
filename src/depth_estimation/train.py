import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DPTForDepthEstimation, DPTImageProcessor
from src.depth_estimation.dataset import SatelliteDepthDataset,get_transforms
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        grad_x_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_y_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        grad_x_target = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_y_target = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = torch.abs(grad_x_pred - grad_x_target).mean()
        loss_y = torch.abs(grad_y_pred - grad_y_target).mean()

        return (loss_x + loss_y) / 2

class DepthEstimationTrainer:
    def __init__(self, model_name="Intel/dpt-large", learning_rate=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Image processor
        self.image_processor = DPTImageProcessor.from_pretrained(model_name)
        
        # Loss function - using a combination of losses
        self.depth_criterion = nn.SmoothL1Loss()
        self.gradient_criterion = GradientLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def compute_gradient_loss(self, pred, target):
        """Compute gradient loss for depth maps"""
        grad_x_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_y_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        grad_x_target = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_y_target = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        loss_x = torch.abs(grad_x_pred - grad_x_target).mean()
        loss_y = torch.abs(grad_y_pred - grad_y_target).mean()
        
        return (loss_x + loss_y) / 2

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, depth_maps) in enumerate(tqdm(dataloader)):
            images = images.to(self.device)
            depth_maps = depth_maps.to(self.device)
            
            # Prepare inputs
            inputs = self.image_processor(images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            # Resize predicted depth to match target
            if predicted_depth.shape[-2:] != depth_maps.shape[-2:]:
                predicted_depth = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=depth_maps.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            # Compute losses
            depth_loss = self.depth_criterion(predicted_depth, depth_maps.squeeze(1))
            gradient_loss = self.compute_gradient_loss(
                predicted_depth.unsqueeze(1), 
                depth_maps
            )
            
            # Combined loss
            loss = depth_loss + 0.5 * gradient_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, depth_maps in tqdm(dataloader):
                images = images.to(self.device)
                depth_maps = depth_maps.to(self.device)
                
                inputs = self.image_processor(images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
                
                # Resize if needed
                if predicted_depth.shape[-2:] != depth_maps.shape[-2:]:
                    predicted_depth = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=depth_maps.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                
                loss = self.depth_criterion(predicted_depth, depth_maps.squeeze(1))
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=50):
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            
            # Training
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f'best_dpt_model.pth')
                print(f'New best model saved with val_loss: {val_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Plot losses
        self.plot_losses(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.savefig('training_losses.png')
        plt.close()

def main():
    # Dataset paths

    train_image_dir = "/content/drive/MyDrive/ProjeDosyalari/anh/train/rgb/"
    train_depth_dir = "/content/drive/MyDrive/ProjeDosyalari/anh/train/dsm/"
    val_image_dir = "/content/drive/MyDrive/ProjeDosyalari/anh/val/rgb/"
    val_depth_dir = "/content/drive/MyDrive/ProjeDosyalari/anh/val/dsm/"
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size=384)
    
    # Create datasets
    train_dataset = SatelliteDepthDataset(
        train_image_dir, train_depth_dir, transform=train_transform
    )
    val_dataset = SatelliteDepthDataset(
        val_image_dir, val_depth_dir, transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    
    # Initialize trainer
    trainer = DepthEstimationTrainer(
        model_name="Intel/dpt-large",  # or "Intel/dpt-hybrid-midas"
        learning_rate=1e-4
    )
    
    # Start training
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, epochs=50
    )

if __name__ == "__main__":
    main()