
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from src.data.milk10k_multimodal_dataset import (
    MILK10kMultimodalDataset,
)
from src.data.paths import MILK10K_IMAGE_ROOT
from src.models.resnet50_multimodal import (
    DermalensResNet50Multimodal,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED = 42

BATCH_SIZE = 32
EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "baseline_c_multimodal_best.pt"

MASTER_INDEX = (
    PROJECT_ROOT / "data" / "metadata" / "milk10k_master_index.csv"
)
SPLIT_CSV = (
    PROJECT_ROOT / "data" / "metadata" / "milk10k_splits.csv"
)
IMAGE_ROOT = MILK10K_IMAGE_ROOT

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CLASS_NAMES = DermalensResNet50Multimodal.CLASS_NAMES


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):
    class_to_index = {
        name: index
        for index, name in enumerate(CLASS_NAMES)
    }

    labels = (
        dataset.data["diagnosis"]
        .map(class_to_index)
        .to_numpy()
    )

    if np.isnan(labels).any():
        raise ValueError(
            "Some training diagnoses do not match CLASS_NAMES."
        )

    labels = labels.astype(int)

    class_counts = np.bincount(
        labels,
        minlength=len(CLASS_NAMES),
    )

    if np.any(class_counts == 0):
        raise ValueError(
            "At least one class has zero training examples."
        )

    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / weights.mean()

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

def create_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.85, 1.0),
            ratio=(0.95, 1.05),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, evaluation_transform


# ============================================================
# DATASETS
# ============================================================

def create_datasets():
    train_transform, evaluation_transform = create_transforms()

    train_dataset = MILK10kMultimodalDataset(
        master_index_csv=str(MASTER_INDEX),
        split_csv=str(SPLIT_CSV),
        split="train",
        image_root=str(IMAGE_ROOT),
        transform=train_transform,
    )

    validation_dataset = MILK10kMultimodalDataset(
        master_index_csv=str(MASTER_INDEX),
        split_csv=str(SPLIT_CSV),
        split="validation",
        image_root=str(IMAGE_ROOT),
        transform=evaluation_transform,
    )

    return train_dataset, validation_dataset


# ============================================================
# DATALOADERS
# ============================================================

def create_loaders():
    train_dataset, validation_dataset = create_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, validation_loader


# ============================================================
# RUN ONE EPOCH
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    all_predictions = []
    all_labels = []

    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(DEVICE)
        metadata = batch["metadata"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images, metadata)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )
        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    if total_samples == 0:
        raise ValueError("The data loader produced no samples.")

    epoch_loss = running_loss / total_samples

    return {
        "loss": epoch_loss,
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            all_labels,
            all_predictions,
        ),
        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),
        "precision": precision_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed()

    # Create output directories only when training is explicitly run.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Dermalens AI — Baseline C Multimodal Training")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(
        "Metadata features: "
        f"{DermalensResNet50Multimodal.METADATA_FEATURES}"
    )
    print(f"Image root: {IMAGE_ROOT}")

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    print("\nCreating multimodal datasets...")

    train_loader, validation_loader = create_loaders()

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")

    # --------------------------------------------------------
    # Verify one batch
    # --------------------------------------------------------

    first_batch = next(iter(train_loader))

    print("\nBatch verification:")
    print(f"Images: {first_batch['image'].shape}")
    print(f"Metadata: {first_batch['metadata'].shape}")
    print(f"Labels: {first_batch['label'].shape}")

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    print("\nCalculating class weights...")

    class_weights = calculate_class_weights(
        train_loader.dataset
    ).to(DEVICE)

    print("Class weights:", class_weights.cpu().numpy())

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading multimodal ResNet-50...")

    model = DermalensResNet50Multimodal(
        pretrained=True
    ).to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    print("\nTraining ready.")

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "-" * 70)
        print(f"Epoch {epoch}/{EPOCHS}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
        )

        scheduler.step(validation_metrics["macro_f1"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train Macro-F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val Loss: {validation_metrics['loss']:.4f}")
        print(
            f"Val Macro-F1: "
            f"{validation_metrics['macro_f1']:.4f}"
        )
        print(
            "Val Balanced Accuracy: "
            f"{validation_metrics['balanced_accuracy']:.4f}"
        )
        print(f"Val Accuracy: {validation_metrics['accuracy']:.4f}")
        print(f"Val Precision: {validation_metrics['precision']:.4f}")
        print(f"Val Recall: {validation_metrics['recall']:.4f}")
        print(f"Learning Rate: {current_lr:.8f}")

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "class_names": model.class_names(),
                    "metadata_features": model.METADATA_FEATURES,
                },
                CHECKPOINT_PATH,
            )

            print(
                "New best model saved "
                f"(Macro-F1: {best_macro_f1:.4f})"
            )
        else:
            epochs_without_improvement += 1
            print("No improvement.")

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("Baseline C training complete.")
    print(f"Best validation Macro-F1: {best_macro_f1:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()