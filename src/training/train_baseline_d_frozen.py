
from pathlib import Path
import random

import numpy as np
import pandas as pd
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
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.milk10k_multimodal_dataset import (
    MILK10kMultimodalDataset,
)
from src.data.paths import MILK10K_IMAGE_ROOT
from src.models.resnet50_multimodal_regularized import (
    DermalensResNet50MultimodalRegularized,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED = 42

MASTER_INDEX = (
    PROJECT_ROOT / "data" / "metadata" / "milk10k_master_index.csv"
)

SPLIT_CSV = (
    PROJECT_ROOT / "data" / "metadata" / "milk10k_splits.csv"
)

IMAGE_ROOT = MILK10K_IMAGE_ROOT

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "baseline_d_frozen_best.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"

BATCH_SIZE = 32
NUM_EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRANSFORMS
# ============================================================

def get_train_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(
            224,
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


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):
    labels = []

    for diagnosis in dataset.data["diagnosis"]:
        labels.append(dataset.CLASS_TO_INDEX[diagnosis])

    labels = np.array(labels)

    num_classes = len(dataset.CLASS_NAMES)

    counts = np.bincount(
        labels,
        minlength=num_classes,
    )

    if np.any(counts == 0):
        raise ValueError(
            "At least one class has zero training examples."
        )

    frequencies = counts / counts.sum()

    # Square-root inverse-frequency weighting.
    weights = 1.0 / np.sqrt(frequencies)

    # Normalize weights to mean 1.
    weights = weights / weights.mean()

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# DATA
# ============================================================

def create_dataloaders():
    train_dataset = MILK10kMultimodalDataset(
        master_index_csv=str(MASTER_INDEX),
        split_csv=str(SPLIT_CSV),
        split="train",
        image_root=str(IMAGE_ROOT),
        transform=get_train_transform(),
    )

    val_dataset = MILK10kMultimodalDataset(
        master_index_csv=str(MASTER_INDEX),
        split_csv=str(SPLIT_CSV),
        split="validation",
        image_root=str(IMAGE_ROOT),
        transform=get_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(targets, predictions):
    return {
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            targets,
            predictions,
        ),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_precision": precision_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        metadata = batch["metadata"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(images, metadata)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(logits, dim=1)

        all_targets.extend(labels.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    return epoch_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        metadata = batch["metadata"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(images, metadata)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(logits, dim=1)

        all_targets.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    return epoch_loss, metrics


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DERMALENS AI — BASELINE D")
    print("Image + Metadata + Partial Backbone Freezing")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Image root: {IMAGE_ROOT}")

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = create_dataloaders()

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(train_dataset)

    print("\nClass weights:")
    print(class_weights.numpy())

    class_weights = class_weights.to(DEVICE)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DermalensResNet50MultimodalRegularized(
        pretrained=True,
        freeze_early_layers=True,
    )

    model = model.to(DEVICE)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"\nTrainable parameters: {trainable_parameters:,}")
    print(f"Total parameters: {total_parameters:,}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
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

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 70)

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Learning rate: {current_lr:.2e}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(
            "Train Balanced Accuracy: "
            f"{train_metrics['balanced_accuracy']:.4f}"
        )
        print(f"Train Macro-F1: {train_metrics['macro_f1']:.4f}")

        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(
            "Val Balanced Accuracy: "
            f"{val_metrics['balanced_accuracy']:.4f}"
        )
        print(f"Val Macro-F1: {val_metrics['macro_f1']:.4f}")

        history.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(val_metrics["macro_f1"])

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                    "class_names": model.class_names(),
                    "config": {
                        "seed": SEED,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "freeze_early_layers": True,
                    },
                },
                CHECKPOINT_PATH,
            )

            print(
                "New best checkpoint saved "
                f"(Val Macro-F1: {best_val_macro_f1:.4f})"
            )
        else:
            epochs_without_improvement += 1
            print("No validation improvement.")

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = REPORT_DIR / "baseline_d_training_history.csv"

    pd.DataFrame(history).to_csv(history_path, index=False)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE D TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()