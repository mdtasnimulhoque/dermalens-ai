
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

from src.data.milk10k_dataset import MILK10kDataset
from src.data.paths import MILK10K_IMAGE_ROOT
from src.models.resnet50_baseline import DermalensResNet50


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_INDEX = PROJECT_ROOT / "data" / "metadata" / "milk10k_master_index.csv"
SPLIT_CSV = PROJECT_ROOT / "data" / "metadata" / "milk10k_splits.csv"

IMAGE_ROOT = MILK10K_IMAGE_ROOT

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports"

BATCH_SIZE = 32
EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.10

EARLY_STOPPING_PATIENCE = 5

SEED = 42


# ============================================================
# DEVICE
# ============================================================

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

    for index in range(len(dataset)):
        sample = dataset[index]
        label = sample["label"]

        if isinstance(label, torch.Tensor):
            label = label.item()

        labels.append(int(label))

    labels = np.asarray(labels, dtype=np.int64)

    num_classes = len(DermalensResNet50.CLASS_NAMES)

    class_counts = np.bincount(
        labels,
        minlength=num_classes,
    )

    if np.any(class_counts == 0):
        missing_classes = np.where(class_counts == 0)[0]

        raise ValueError(
            "One or more classes have zero training samples: "
            f"{missing_classes.tolist()}"
        )

    class_weights = 1.0 / np.sqrt(class_counts)
    class_weights = class_weights / class_weights.mean()

    return torch.tensor(
        class_weights,
        dtype=torch.float32,
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
# TRAIN / VALIDATE
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_targets = []
    all_predictions = []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(logits, dim=1)

        all_targets.extend(
            labels.detach().cpu().numpy()
        )
        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

    average_loss = total_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    return average_loss, metrics


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    print("=" * 70)
    print("DERMALENS AI — BASELINE E")
    print(
        "Image + Augmentation + "
        "Class Balancing + Label Smoothing"
    )
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    print(f"Image root: {IMAGE_ROOT}")

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = MILK10kDataset(
        master_index_csv=str(MASTER_INDEX),
        split_csv=str(SPLIT_CSV),
        split="train",
        image_root=str(IMAGE_ROOT),
        transform=get_train_transform(),
    )

    val_dataset = MILK10kDataset(
        master_index_csv=str(MASTER_INDEX),
        split_csv=str(SPLIT_CSV),
        split="validation",
        image_root=str(IMAGE_ROOT),
        transform=get_eval_transform(),
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

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

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(train_dataset)

    print("\nClass weights:")
    print(class_weights.numpy())

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = DermalensResNet50(pretrained=True)
    model = model.to(DEVICE)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE),
        label_smoothing=LABEL_SMOOTHING,
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # CHECKPOINTING
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR / "baseline_e_label_smoothing_best.pt"
    )
    history_path = REPORT_DIR / "baseline_e_training_history.csv"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 70)
        print(f"Learning rate: {current_lr:.2e}")

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics[
                "balanced_accuracy"
            ],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics[
                "balanced_accuracy"
            ],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        # ----------------------------------------------------
        # LR SCHEDULER
        # ----------------------------------------------------

        scheduler.step(val_metrics["macro_f1"])

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                    "label_smoothing": LABEL_SMOOTHING,
                    "class_weights": class_weights.cpu(),
                    "seed": SEED,
                },
                checkpoint_path,
            )

            print(
                "New best checkpoint saved "
                f"(Val Macro-F1: {best_val_macro_f1:.4f})"
            )
        else:
            epochs_without_improvement += 1
            print("No validation improvement.")

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE E TRAINING COMPLETE")
    print("=" * 70)

    print(f"Best validation Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()