
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
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.milk10k_dataset import MILK10kDataset
from src.data.paths import MILK10K_IMAGE_ROOT
from src.models.resnet50_baseline import DermalensResNet50


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
    / "baseline_e_label_smoothing_best.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"

BATCH_SIZE = 32

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CLASS_NAMES = DermalensResNet50.CLASS_NAMES


# ============================================================
# TRANSFORMS
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
    print("DERMALENS AI — BASELINE E TEST EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Image root: {IMAGE_ROOT}")

    test_dataset = MILK10kDataset(
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

    model = DermalensResNet50(pretrained=False)
    model = model.to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(
        "Best validation Macro-F1: "
        f"{checkpoint['best_val_macro_f1']:.4f}"
    )

    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_lesion_ids = []
    all_image_ids = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_lesion_ids.extend(batch["lesion_id"])
            all_image_ids.extend(batch["image_id"])

    targets = np.asarray(all_targets, dtype=np.int64)
    predictions = np.asarray(all_predictions, dtype=np.int64)
    probabilities = np.asarray(all_probabilities, dtype=np.float32)

    accuracy = accuracy_score(targets, predictions)
    balanced_accuracy = balanced_accuracy_score(targets, predictions)

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("BASELINE E — TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
    print(f"Macro-F1:           {macro_f1:.4f}")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")

    print("\nPer-class results:\n")

    report = classification_report(
        targets,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    print(report)

    confusion = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    predictions_df = pd.DataFrame({
        "lesion_id": all_lesion_ids,
        "image_id": all_image_ids,
        "true_label": targets,
        "true_diagnosis": [
            CLASS_NAMES[index] for index in targets
        ],
        "predicted_label": predictions,
        "predicted_diagnosis": [
            CLASS_NAMES[index] for index in predictions
        ],
        "confidence": probabilities.max(axis=1),
        "correct": targets == predictions,
    })

    for index, class_name in enumerate(CLASS_NAMES):
        predictions_df[f"prob_{class_name}"] = probabilities[:, index]

    metrics_path = REPORT_DIR / "baseline_e_test_metrics.txt"
    confusion_path = REPORT_DIR / "baseline_e_test_confusion_matrix.csv"
    predictions_path = REPORT_DIR / "baseline_e_test_predictions.csv"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("DERMALENS AI — BASELINE E TEST RESULTS\n")
        file.write("=" * 60 + "\n")
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
        file.write("Per-class classification report:\n")
        file.write(report)

    confusion_df.to_csv(confusion_path)
    predictions_df.to_csv(predictions_path, index=False)

    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)
    print(f"Metrics:      {metrics_path}")
    print(f"Confusion:    {confusion_path}")
    print(f"Predictions:  {predictions_path}")


if __name__ == "__main__":
    main()