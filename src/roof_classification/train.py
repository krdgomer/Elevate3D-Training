import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import copy
from collections import Counter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class for CSV-based labels
class RoofCSVDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load CSV file
        csv_path = os.path.join(root_dir, csv_file)
        self.data_df = pd.read_csv(csv_path)
        
        # Clean column names (remove any leading/trailing spaces)
        self.data_df.columns = self.data_df.columns.str.strip()
        
        # Find the actual class columns (exclude filename and Unlabeled)
        all_columns = list(self.data_df.columns)
        if 'filename' in all_columns:
            all_columns.remove('filename')
        if 'Unlabeled' in all_columns:
            all_columns.remove('Unlabeled')
        
        self.class_columns = all_columns
        self.class_names = self.class_columns
        
        # Store image paths and labels
        self.image_paths = []
        self.labels = []
        
        for idx in range(len(self.data_df)):
            img_name = self.data_df.iloc[idx]['filename'].strip()
            img_path = os.path.join(self.root_dir, img_name)
            
            # Get label (one-hot encoded)
            label_values = self.data_df.iloc[idx][self.class_columns].values.astype(np.float32)
            label = np.argmax(label_values)
            
            self.image_paths.append(img_path)
            self.labels.append(label)
        
        print(f"Found class columns: {self.class_columns}")
        print(f"Dataset size: {len(self.data_df)}")
        print(f"Class distribution: {Counter(self.labels)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Enhanced data transformations with more augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(235, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(235),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(235),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create datasets and dataloaders with class balancing
def create_dataloaders(data_dir, batch_size=16):
    # Define paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Create datasets
    train_dataset = RoofCSVDataset(train_dir, '_classes.csv', transform=data_transforms['train'])
    val_dataset = RoofCSVDataset(val_dir, '_classes.csv', transform=data_transforms['val'])
    test_dataset = RoofCSVDataset(test_dir, '_classes.csv', transform=data_transforms['test'])
    
    # Calculate class weights for weighted sampling
    class_counts = Counter(train_dataset.labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, train_dataset.class_names, class_counts

# Improved model definition with EfficientNet
def create_model(num_classes=5, pretrained=True):
    # Use EfficientNet which often works better with limited data
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Freeze early layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last few layers
    for param in model.features[-3:].parameters():  # Unfreeze last 3 layers
        param.requires_grad = True
    
    # Replace the classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    return model.to(device)

# Enhanced training function with early stopping and learning rate scheduling
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 10
    no_improve = 0
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_accuracy = running_corrects.double() / len(val_loader.dataset)
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy.cpu())
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Early stopping and model checkpointing
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
            print(f"New best model saved with accuracy: {best_acc:.4f}")
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        # Unfreeze more layers halfway through training
        if epoch == num_epochs // 2:
            for param in model.parameters():
                param.requires_grad = True
            print("All layers unfrozen for fine-tuning")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, val_accuracies

# Enhanced evaluation function with better metrics
def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Calculate per-class accuracy
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    for i in range(len(all_labels)):
        label = all_labels[i]
        class_total[label] += 1
        if all_labels[i] == all_preds[i]:
            class_correct[label] += 1
    
    print("Per-class accuracy:")
    for i, cls in enumerate(class_names):
        if class_total[i] > 0:
            print(f"{cls}: {class_correct[i] / class_total[i]:.3f} ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"{cls}: No samples")
    
    return all_preds, all_labels, all_probs

# Plot training history with more details
def plot_training_history(train_losses, val_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(val_accuracies, linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Add best accuracy annotation
    best_acc = max(val_accuracies)
    best_epoch = val_accuracies.index(best_acc) + 1
    ax2.annotate(f'Best: {best_acc:.3f} at epoch {best_epoch}', 
                xy=(best_epoch-1, best_acc),
                xytext=(best_epoch-1, best_acc-0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set your data directory path here
    data_dir = "/content/drive/MyDrive/ProjeDosyalari/roof/"  # Update this path
    
    # Hyperparameters
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.0005
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names, class_counts = create_dataloaders(data_dir, batch_size)
    
    print(f"Dataset classes: {class_names}")
    print(f"Class counts: {class_counts}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Calculate class weights for loss function
    class_weights = [1.0 / class_counts[i] for i in range(len(class_names))]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Create model
    model = create_model(num_classes=len(class_names))
    
    # Optimizer
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': learning_rate, 'weight_decay': 0.01}
    ])
    
    # Train the model
    print("Starting training...")
    model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    # Evaluate the model on test set
    print("Evaluating model on test set...")
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, class_names)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'val_accuracy': max(val_accuracies)
    }, 'roof_classifier_improved.pth')
    print("Model saved as roof_classifier_improved.pth")
    
    # Print final summary
    print("\n=== TRAINING SUMMARY ===")
    print(f"Best validation accuracy: {max(val_accuracies):.4f}")
    print(f"Final test accuracy: {np.mean(np.array(all_preds) == np.array(all_labels)):.4f}")