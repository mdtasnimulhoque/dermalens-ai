
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms
from torch.utils.data import DataLoader

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

BATCH_SIZE = 32

MASTER_INDEX_CSV = (
    PROJECT_ROOT / "data" / "metadata" / "milk10k_master_index.csv"
)

SPLIT_CSV = (
    PROJECT_ROOT / "data" / "metadata" / "milk10k_splits.csv"
)

IMAGE_ROOT = MILK10K_IMAGE_ROOT

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "baseline_c_multimodal_best.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"

PREDICTIONS_PATH = REPORT_DIR / "baseline_c_test_predictions.csv"
METRICS_PATH = REPORT_DIR / "baseline_c_test_metrics.txt"
CONFUSION_MATRIX_PATH = (
    REPORT_DIR / "baseline_c_test_confusion_matrix.csv"
)


# ============================================================
# TRANSFORM
# ============================================================

def create_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# CREATE TEST LOADER
# ============================================================

def create_test_loader():
    test_dataset = MILK10kMultimodalDataset(
        master_index_csv=str(MASTER_INDEX_CSV),
        split_csv=str(SPLIT_CSV),
        split="test",
        image_root=str(IMAGE_ROOT),
        transform=create_test_transform(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return test_dataset, test_loader


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = DermalensResNet50Multimodal(pretrained=False)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):
    all_labels = []
    all_predictions = []
    all_probabilities = []

    lesion_ids = []
    image_ids = []
    diagnoses = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            metadata = batch["metadata"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(images, metadata)

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            lesion_ids.extend(batch["lesion_id"])
            image_ids.extend(batch["image_id"])
            diagnoses.extend(batch["diagnosis"])

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
        lesion_ids,
        image_ids,
        diagnoses,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Dermalens AI — Baseline C Test Evaluation")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading held-out test dataset...")

    test_dataset, test_loader = create_test_loader()

    print(f"Test images: {len(test_dataset)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading best Baseline C checkpoint...")

    model, checkpoint = load_model()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        f"Best validation Macro-F1: "
        f"{checkpoint['best_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nEvaluating on the held-out test set...")

    (
        labels,
        predictions,
        probabilities,
        lesion_ids,
        image_ids,
        diagnoses,
    ) = evaluate(model, test_loader)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(labels, predictions)

    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Balanced Accuracy:    {balanced_accuracy:.4f}")
    print(f"Macro-F1:             {macro_f1:.4f}")
    print(f"Macro Precision:      {macro_precision:.4f}")
    print(f"Macro Recall:         {macro_recall:.4f}")

    # --------------------------------------------------------
    # Per-class report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    class_names = DermalensResNet50Multimodal.CLASS_NAMES

    report = classification_report(
        labels,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(len(class_names)),
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    # --------------------------------------------------------
    # Prediction CSV
    # --------------------------------------------------------

    predicted_names = [
        class_names[int(prediction)]
        for prediction in predictions
    ]

    true_names = [
        class_names[int(label)]
        for label in labels
    ]

    confidence = probabilities.max(axis=1)

    prediction_df = pd.DataFrame({
        "lesion_id": lesion_ids,
        "image_id": image_ids,
        "true_diagnosis": true_names,
        "predicted_diagnosis": predicted_names,
        "confidence": confidence,
    })

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cm_df.to_csv(CONFUSION_MATRIX_PATH, index=True)
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    metrics_text = f"""
Dermalens AI — Baseline C
Held-Out Test Evaluation

Checkpoint:
{CHECKPOINT_PATH}

Best validation epoch:
{checkpoint["epoch"]}

Best validation Macro-F1:
{checkpoint["best_macro_f1"]:.4f}

Test Accuracy:
{accuracy:.4f}

Test Balanced Accuracy:
{balanced_accuracy:.4f}

Test Macro-F1:
{macro_f1:.4f}

Test Macro Precision:
{macro_precision:.4f}

Test Macro Recall:
{macro_recall:.4f}

Per-Class Results:

{report}
"""

    METRICS_PATH.write_text(
        metrics_text.strip() + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("=" * 70)
    print("Evaluation complete.")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()