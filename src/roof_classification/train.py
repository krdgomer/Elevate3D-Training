import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RoofDataset, train_transform, test_transform
from model import build_model

import os

# Paths
train_dir = "/content/drive/MyDrive/ProjeDosyalari/roof/train"
val_dir   = "/content/drive/MyDrive/ProjeDosyalari/roof/val"

batch_size = 32
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
train_dataset = RoofDataset(train_dir, transform=train_transform)
val_dataset   = RoofDataset(val_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# Model
model = build_model(num_classes, freeze_backbone=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0, 0

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

        # Validation
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_roof_model.pth")

    print("Training complete. Best Val Acc:", best_acc)

train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)
