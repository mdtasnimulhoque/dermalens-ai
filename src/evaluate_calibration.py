from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

from src.data.milk10k_dataloader import create_dataloaders
from src.models.resnet50_baseline import DermalensResNet50


CHECKPOINT = "models/checkpoints/baseline_a_best.pt"
BATCH_SIZE = 32
NUM_BINS = 10

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def calculate_ece(confidences, correct, num_bins=10):

    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    ece = 0.0

    for i in range(num_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == num_bins - 1:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )

        if not np.any(mask):
            continue

        bin_confidence = confidences[mask].mean()
        bin_accuracy = correct[mask].mean()
        bin_fraction = mask.mean()

        ece += (
            bin_fraction
            * abs(bin_accuracy - bin_confidence)
        )

    return ece


def main():

    print("=" * 70)
    print("Dermalens AI — Baseline A Calibration Analysis")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

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

    all_confidences = []
    all_correct = []
    all_probabilities = []
    all_labels = []

    print("\nRunning calibration evaluation...")

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            confidences = torch.max(
                probabilities,
                dim=1,
            ).values

            correct = (
                predictions == labels
            )

            all_confidences.extend(
                confidences.cpu().numpy()
            )

            all_correct.extend(
                correct.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    confidences = np.array(all_confidences)
    correct = np.array(all_correct)

    probabilities = np.array(
        all_probabilities
    )

    labels = np.array(all_labels)

    ece = calculate_ece(
        confidences,
        correct,
        NUM_BINS,
    )

    # Multiclass Brier score.
    one_hot_labels = np.eye(
        DermalensResNet50.NUM_CLASSES
    )[labels]

    brier_score = np.mean(
        np.sum(
            (
                probabilities
                - one_hot_labels
            ) ** 2,
            axis=1,
        )
    )

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    print(
        f"Expected Calibration Error: "
        f"{ece:.4f}"
    )

    print(
        f"Multiclass Brier Score: "
        f"{brier_score:.4f}"
    )

    print(
        f"Mean Confidence: "
        f"{confidences.mean():.4f}"
    )

    print(
        f"Actual Accuracy: "
        f"{correct.mean():.4f}"
    )

    print("\n" + "=" * 70)
    print("RELIABILITY TABLE")
    print("=" * 70)

    bin_edges = np.linspace(
        0.0,
        1.0,
        NUM_BINS + 1,
    )

    rows = []

    for i in range(NUM_BINS):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == NUM_BINS - 1:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )

        if not np.any(mask):
            continue

        rows.append(
            {
                "confidence_range":
                    f"{lower:.1f}-{upper:.1f}",
                "samples":
                    int(mask.sum()),
                "mean_confidence":
                    confidences[mask].mean(),
                "accuracy":
                    correct[mask].mean(),
            }
        )

    reliability = pd.DataFrame(rows)

    print(
        reliability.to_string(
            index=False
        )
    )

    Path("reports").mkdir(
        parents=True,
        exist_ok=True,
    )

    reliability.to_csv(
        "reports/baseline_a_reliability.csv",
        index=False,
    )

    # Reliability diagram.

    plt.figure(figsize=(7, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        reliability["mean_confidence"],
        reliability["accuracy"],
        marker="o",
        label="Baseline A",
    )

    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title(
        "Baseline A Reliability Diagram"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "reports/baseline_a_reliability.png",
        dpi=200,
    )

    plt.close()

    print(
        "\nSaved:"
        "\n- reports/baseline_a_reliability.csv"
        "\n- reports/baseline_a_reliability.png"
    )

    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()