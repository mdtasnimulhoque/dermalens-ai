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

from src.data.milk10k_dataloader import create_dataloaders
from src.models.resnet50_baseline import DermalensResNet50


SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def calculate_class_weights(dataset):
    """
    Calculate square-root inverse-frequency class weights
    using the training dataset only.

    This reduces the influence of extremely rare classes
    compared with ordinary inverse-frequency weighting.
    """

    class_to_index = {
        name: index
        for index, name in enumerate(
            DermalensResNet50.CLASS_NAMES
        )
    }

    labels = dataset.data["diagnosis"].map(
        class_to_index
    ).to_numpy()

    class_counts = np.bincount(
        labels,
        minlength=DermalensResNet50.NUM_CLASSES,
    )

    weights = 1.0 / np.sqrt(class_counts)

    weights = weights / weights.mean()

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
):
    """
    Run one training or validation epoch.
    """

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
        labels = batch["label"].to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            if training:
                loss.backward()
                optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss / total_samples
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions,
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "loss": epoch_loss,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def main():

    set_seed()

    print("=" * 70)
    print("Dermalens AI — Baseline A Training")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    print("\nLoading datasets...")

    train_loader, validation_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )

    print("\nCalculating class weights...")

    class_weights = calculate_class_weights(
        train_loader.dataset
    ).to(DEVICE)

    print(
        "Class weights:",
        class_weights.detach().cpu().numpy(),
    )

    print("\nLoading pretrained ResNet-50...")

    model = DermalensResNet50(
        pretrained=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

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

    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    best_checkpoint = (
        CHECKPOINT_DIR
        / "baseline_a_best.pt"
    )

    print("\nTraining ready.")

    print(
        f"Best checkpoint: "
        f"{best_checkpoint}"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print("\n" + "-" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

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

        scheduler.step(
            validation_metrics["macro_f1"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Train Loss: "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Train Macro-F1: "
            f"{train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val Loss: "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            f"Val Macro-F1: "
            f"{validation_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val Balanced Accuracy: "
            f"{validation_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.8f}"
        )

        if (
            validation_metrics["macro_f1"]
            > best_macro_f1
        ):

            best_macro_f1 = (
                validation_metrics["macro_f1"]
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "class_names": model.class_names(),
                },
                best_checkpoint,
            )

            print(
                f"✓ New best model saved "
                f"(Macro-F1: {best_macro_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

            print("No improvement.")

        if epochs_without_improvement >= 5:

            print(
                "\nEarly stopping triggered."
            )

            break

    print("\n" + "=" * 70)

    print("Training complete.")

    print(
        f"Best validation Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{best_checkpoint}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()