
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_INDEX = (
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
    / "baseline_d_frozen_best.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"

BATCH_SIZE = 32

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ============================================================
# EVALUATION TRANSFORM
# ============================================================

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
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("DERMALENS AI — BASELINE D TEST EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = MILK10kMultimodalDataset(
        master_index_csv=str(MASTER_INDEX),
        split_csv=str(SPLIT_CSV),
        split="test",
        image_root=str(IMAGE_ROOT),
        transform=get_eval_transform(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Test images: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DermalensResNet50MultimodalRegularized(
        pretrained=False,
        freeze_early_layers=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(
        "Best validation Macro-F1: "
        f"{checkpoint['best_val_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_lesion_ids = []
    all_image_ids = []
    all_diagnoses = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            metadata = batch["metadata"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(images, metadata)

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_lesion_ids.extend(batch["lesion_id"])
            all_image_ids.extend(batch["image_id"])
            all_diagnoses.extend(batch["diagnosis"])

    all_targets = np.asarray(all_targets)
    all_predictions = np.asarray(all_predictions)
    all_probabilities = np.asarray(all_probabilities)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(all_targets, all_predictions)
    balanced_accuracy = balanced_accuracy_score(
        all_targets,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    print("\nOverall test results")
    print("-" * 70)
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Balanced Accuracy:    {balanced_accuracy:.4f}")
    print(f"Macro-F1:             {macro_f1:.4f}")
    print(f"Macro Precision:      {macro_precision:.4f}")
    print(f"Macro Recall:         {macro_recall:.4f}")

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    class_names = model.class_names()

    per_class_f1 = f1_score(
        all_targets,
        all_predictions,
        average=None,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    per_class_precision = precision_score(
        all_targets,
        all_predictions,
        average=None,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    per_class_recall = recall_score(
        all_targets,
        all_predictions,
        average=None,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    support = np.bincount(
        all_targets,
        minlength=len(class_names),
    )

    per_class_df = pd.DataFrame({
        "class": class_names,
        "precision": per_class_precision,
        "recall": per_class_recall,
        "f1": per_class_f1,
        "support": support,
    })

    print("\nPer-class results")
    print("-" * 70)
    print(per_class_df.to_string(index=False))

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_targets,
        all_predictions,
        labels=np.arange(len(class_names)),
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    # --------------------------------------------------------
    # Prediction confidence
    # --------------------------------------------------------

    max_probabilities = all_probabilities.max(axis=1)
    correct_mask = all_predictions == all_targets

    mean_confidence_correct = (
        max_probabilities[correct_mask].mean()
        if correct_mask.any()
        else float("nan")
    )

    mean_confidence_incorrect = (
        max_probabilities[~correct_mask].mean()
        if (~correct_mask).any()
        else float("nan")
    )

    print("\nConfidence analysis")
    print("-" * 70)
    print(
        f"Mean confidence — correct:   "
        f"{mean_confidence_correct:.4f}"
    )
    print(
        f"Mean confidence — incorrect: "
        f"{mean_confidence_incorrect:.4f}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_rows = []

    for i in range(len(all_targets)):
        row = {
            "lesion_id": all_lesion_ids[i],
            "image_id": all_image_ids[i],
            "true_diagnosis": all_diagnoses[i],
            "true_index": int(all_targets[i]),
            "predicted_index": int(all_predictions[i]),
            "predicted_diagnosis": class_names[all_predictions[i]],
            "confidence": float(max_probabilities[i]),
            "correct": bool(correct_mask[i]),
        }

        for class_index, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(
                all_probabilities[i, class_index]
            )

        prediction_rows.append(row)

    predictions_df = pd.DataFrame(prediction_rows)

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORT_DIR / "baseline_d_test_metrics.txt"
    predictions_path = REPORT_DIR / "baseline_d_test_predictions.csv"
    confusion_path = REPORT_DIR / "baseline_d_test_confusion_matrix.csv"

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("DERMALENS AI — BASELINE D TEST RESULTS\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Test images: {len(test_dataset)}\n")
        file.write(f"Checkpoint epoch: {checkpoint['epoch']}\n")
        file.write(
            "Best validation Macro-F1: "
            f"{checkpoint['best_val_macro_f1']:.4f}\n\n"
        )

        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
        file.write(f"Macro-F1: {macro_f1:.4f}\n")
        file.write(f"Macro Precision: {macro_precision:.4f}\n")
        file.write(f"Macro Recall: {macro_recall:.4f}\n\n")

        file.write("Confidence:\n")
        file.write(
            f"Correct predictions: {mean_confidence_correct:.4f}\n"
        )
        file.write(
            f"Incorrect predictions: {mean_confidence_incorrect:.4f}\n\n"
        )

        file.write("Per-class metrics:\n")
        file.write(per_class_df.to_string(index=False))
        file.write("\n")

    predictions_df.to_csv(predictions_path, index=False)
    cm_df.to_csv(confusion_path)

    print("\nReports saved:")
    print(metrics_path)
    print(predictions_path)
    print(confusion_path)

    print("\n" + "=" * 70)
    print("BASELINE D TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()