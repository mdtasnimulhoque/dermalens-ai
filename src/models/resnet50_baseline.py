import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class DermalensResNet50(nn.Module):
    """
    Dermalens AI — Baseline A

    Clinical image-only skin-lesion classification model.

    Backbone:
        ImageNet-pretrained ResNet-50

    Output:
        11 MILK10k diagnostic categories

    This model produces classification logits.
    Probabilities and uncertainty are handled separately.
    """

    CLASS_NAMES = [
        "AKIEC",
        "BCC",
        "BEN_OTH",
        "BKL",
        "DF",
        "INF",
        "MAL_OTH",
        "MEL",
        "NV",
        "SCCKA",
        "VASC",
    ]

    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet50(weights=weights)

        # Replace ImageNet's 1000-class classifier
        # with MILK10k's 11-class classifier.
        input_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            input_features,
            self.NUM_CLASSES,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return raw classification logits.

        Shape:
            input  -> [batch_size, 3, 224, 224]
            output -> [batch_size, 11]
        """
        return self.backbone(x)

    @classmethod
    def class_names(cls):
        return cls.CLASS_NAMES.copy()
