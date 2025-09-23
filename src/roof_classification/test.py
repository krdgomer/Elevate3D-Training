import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src.roof_classification.dataset import RoofDataset, test_transform
from src.roof_classification.model import build_model

# Paths
test_dir = "data/test"

batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
test_dataset = RoofDataset(test_dir, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Model
num_classes = len(test_dataset.classes)
model = build_model(num_classes, freeze_backbone=True).to(device)
model.load_state_dict(torch.load("best_roof_model.pth"))
model.eval()

# Evaluation
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
