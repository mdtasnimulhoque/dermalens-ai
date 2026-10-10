import numpy as np
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

from src.data.milk10k_dataloader import create_dataloaders
from src.models.resnet50_baseline import DermalensResNet50


CHECKPOINT = "models/checkpoints/baseline_b_augmented_best.pt"
BATCH_SIZE = 32

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def main():

    print("=" * 70)
    print("Dermalens AI — Baseline B Test Evaluation")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")

    print("\nLoading test dataset...")

    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    print("\nLoading best Baseline B checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model = DermalensResNet50(
        pretrained=False
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation Macro-F1: "
        f"{checkpoint['best_macro_f1']:.4f}"
    )

    all_predictions = []
    all_labels = []

    print("\nRunning test evaluation...")

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(images)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    print("\n" + "=" * 70)
    print("BASELINE B — OVERALL TEST RESULTS")
    print("=" * 70)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Balanced Accuracy:   {balanced_accuracy:.4f}")
    print(f"Macro-F1:            {macro_f1:.4f}")
    print(f"Macro Precision:     {macro_precision:.4f}")
    print(f"Macro Recall:        {macro_recall:.4f}")

    print("\n" + "=" * 70)
    print("BASELINE B — PER-CLASS RESULTS")
    print("=" * 70)

    class_names = DermalensResNet50.CLASS_NAMES

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    print(report)

    print("=" * 70)
    print("BASELINE B — CONFUSION MATRIX")
    print("=" * 70)

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
    )

    print("\nRows = True class")
    print("Columns = Predicted class\n")

    print("        " + " ".join(
        f"{name:>8}"
        for name in class_names
    ))

    for index, row in enumerate(matrix):

        print(
            f"{class_names[index]:>8} "
            + " ".join(
                f"{value:8d}"
                for value in row
            )
        )

    print("\n" + "=" * 70)
    print("BASELINE B TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()