import torch
import torch.nn as nn
import timm


class MelanomaClassifier(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b3", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        # Load pretrained backbone without the classification head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.dropout  = nn.Dropout(dropout)
        # Binary output — sigmoid applied in loss function (BCEWithLogitsLoss)
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        return self.head(features)


class EnsembleClassifier(nn.Module):
    """Average predictions from multiple trained models."""
    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [torch.sigmoid(m(x)) for m in self.models]
        return torch.stack(outputs).mean(dim=0)
