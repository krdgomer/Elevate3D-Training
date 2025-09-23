import torch.nn as nn
from torchvision import models

def build_model(num_classes, freeze_backbone=True):
    model = models.resnet50(pretrained=True)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model
