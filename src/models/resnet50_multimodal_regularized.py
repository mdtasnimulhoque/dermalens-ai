import torch
import torch.nn as nn

from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)


class DermalensResNet50MultimodalRegularized(
    nn.Module
):
    """
    Baseline D:
    Clinical image + structured metadata
    with a partially frozen ResNet-50 backbone.

    Image branch:
        ResNet-50 → 2048-dimensional representation

    Metadata branch:
        18 features → 64-dimensional representation

    Fusion:
        2048 + 64 → 256 → 11 classes
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

    def __init__(
        self,
        pretrained=True,
        freeze_early_layers=True,
    ):

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

        # Remove original ImageNet classifier.
        self.image_backbone.fc = nn.Identity()

        # ----------------------------------------------------
        # Partial freezing
        #
        # Freeze:
        #   conv1
        #   bn1
        #   layer1
        #   layer2
        #
        # Keep trainable:
        #   layer3
        #   layer4
        #
        # This allows higher-level visual features to adapt
        # while reducing the number of parameters that can
        # overfit the relatively small dataset.
        # ----------------------------------------------------

        if freeze_early_layers:

            for parameter in (
                self.image_backbone.conv1.parameters()
            ):
                parameter.requires_grad = False

            for parameter in (
                self.image_backbone.bn1.parameters()
            ):
                parameter.requires_grad = False

            for parameter in (
                self.image_backbone.layer1.parameters()
            ):
                parameter.requires_grad = False

            for parameter in (
                self.image_backbone.layer2.parameters()
            ):
                parameter.requires_grad = False

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
                p=0.30
            ),

            nn.Linear(
                64,
                self.METADATA_HIDDEN,
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.20
            ),
        )

        # ====================================================
        # FUSION CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                self.IMAGE_FEATURES
                + self.METADATA_HIDDEN,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.50
            ),

            nn.Linear(
                256,
                self.NUM_CLASSES,
            ),
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        image,
        metadata,
    ):

        image_features = (
            self.image_backbone(image)
        )

        metadata_features = (
            self.metadata_encoder(
                metadata
            )
        )

        fused_features = torch.cat(
            [
                image_features,
                metadata_features,
            ],
            dim=1,
        )

        logits = self.classifier(
            fused_features
        )

        return logits

    # ========================================================
    # CLASS NAMES
    # ========================================================

    @classmethod
    def class_names(cls):

        return cls.CLASS_NAMES.copy()