import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import copy

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
        
        print(f"Found class columns: {self.class_columns}")
        print(f"Dataset size: {len(self.data_df)}")
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.data_df.iloc[idx]['filename'].strip()
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label (one-hot encoded)
        label_values = self.data_df.iloc[idx][self.class_columns].values.astype(np.float32)
        label = np.argmax(label_values)  # Convert one-hot to class index
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data transformations with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(235),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

# Create datasets and dataloaders
def create_dataloaders(data_dir, batch_size=32):
    # Define paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    # Create datasets
    train_dataset = RoofCSVDataset(train_dir, '_classes.csv', transform=data_transforms['train'])
    val_dataset = RoofCSVDataset(val_dir, '_classes.csv', transform=data_transforms['val'])
    test_dataset = RoofCSVDataset(test_dir, '_classes.csv', transform=data_transforms['test'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Set workers to 0 to avoid issues
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, train_dataset.class_names

# Model definition
def create_model(num_classes=5, pretrained=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model.to(device)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
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
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Deep copy the best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, val_accuracies

# Evaluation function
def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    return all_preds, all_labels

# Plot training history
def plot_training_history(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.show()

# Check class distribution
def check_class_distribution(loader, class_names):
    class_counts = {cls: 0 for cls in class_names}
    for _, labels in loader:
        for label in labels:
            class_counts[class_names[label]] += 1
    return class_counts

# Main execution
if __name__ == "__main__":
    # Set your data directory path here
    data_dir = "/content/drive/MyDrive/ProjeDosyalari/roof/"  # Update this path
    
    # Hyperparameters
    batch_size = 16
    num_epochs = 30
    learning_rate = 0.0005
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(data_dir, batch_size)
    
    print(f"Dataset classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Check class distribution
    train_counts = check_class_distribution(train_loader, class_names)
    val_counts = check_class_distribution(val_loader, class_names)
    test_counts = check_class_distribution(test_loader, class_names)
    
    print("Training set class distribution:", train_counts)
    print("Validation set class distribution:", val_counts)
    print("Test set class distribution:", test_counts)
    
    # Create model
    model = create_model(num_classes=len(class_names))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print("Starting training...")
    model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    # Evaluate the model on test set
    print("Evaluating model on test set...")
    evaluate_model(model, test_loader, class_names)
    
    # Save the model
    torch.save(model.state_dict(), 'roof_classifier.pth')
    print("Model saved as roof_classifier.pth")