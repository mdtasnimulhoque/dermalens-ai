import torch
import torch.nn as nn

from torchvision.models import ResNet50_Weights, resnet50


class DermalensResNet50Multimodal(nn.Module):
    """
    Baseline C: Clinical image + structured metadata.

    Image branch:
        ResNet-50 → 2048-dimensional representation

    Metadata branch:
        18 metadata features → small MLP → 64-dimensional representation

    Fusion:
        2048 image features + 64 metadata features
        → classification head → 11 classes
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

    IMAGE_FEATURES = 2048
    METADATA_FEATURES = 18
    METADATA_HIDDEN = 64

    def __init__(self, pretrained=True):

        super().__init__()

        # ====================================================
        # IMAGE BRANCH
        # ====================================================

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        self.image_backbone = resnet50(
            weights=weights
        )

        # Remove the original ImageNet classifier.
        self.image_backbone.fc = nn.Identity()

        # ====================================================
        # METADATA BRANCH
        # ====================================================

        self.metadata_encoder = nn.Sequential(
            nn.Linear(
                self.METADATA_FEATURES,
                64,
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                64
            ),

            nn.Dropout(
                p=0.2
            ),

            nn.Linear(
                64,
                self.METADATA_HIDDEN,
            ),

            nn.ReLU(),
        )

        # ====================================================
        # FUSION CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(
            nn.Linear(
                self.IMAGE_FEATURES
                + self.METADATA_HIDDEN,
                512,
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.4
            ),

            nn.Linear(
                512,
                self.NUM_CLASSES,
            ),
        )

    def forward(
        self,
        image,
        metadata,
    ):

        # Image representation
        image_features = self.image_backbone(
            image
        )

        # Metadata representation
        metadata_features = self.metadata_encoder(
            metadata
        )

        # Combine both modalities
        fused_features = torch.cat(
            [
                image_features,
                metadata_features,
            ],
            dim=1,
        )

        # Final prediction
        logits = self.classifier(
            fused_features
        )

        return logits

    @classmethod
    def class_names(cls):
        return cls.CLASS_NAMES.copy()