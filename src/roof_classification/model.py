import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    model = models.resnet50(pretrained=True)

    # Freeze everything except layer4 + fc
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model
