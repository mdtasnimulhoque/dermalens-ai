from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from src.data.milk10k_dataloader import create_dataloaders
from src.models.resnet50_baseline import DermalensResNet50


CHECKPOINT = "models/checkpoints/baseline_a_best.pt"
BATCH_SIZE = 32
NUM_BINS = 10

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def calculate_ece(
    probabilities,
    labels,
    num_bins=10,
):
    confidences, predictions = torch.max(
        probabilities,
        dim=1,
    )

    correct = (
        predictions == labels
    )

    confidences = confidences.cpu().numpy()
    correct = correct.cpu().numpy()

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
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return ece


def calculate_brier_score(
    probabilities,
    labels,
    num_classes,
):
    one_hot = np.eye(
        num_classes
    )[labels.cpu().numpy()]

    probabilities_np = (
        probabilities.cpu().numpy()
    )

    return np.mean(
        np.sum(
            (
                probabilities_np
                - one_hot
            ) ** 2,
            axis=1,
        )
    )


def collect_logits(
    model,
    loader,
):
    all_logits = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(images)

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    return (
        torch.cat(all_logits),
        torch.cat(all_labels),
    )


def find_temperature(
    validation_logits,
    validation_labels,
):
    temperature = nn.Parameter(
        torch.ones(1)
    )

    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.01,
        max_iter=100,
    )

    criterion = nn.CrossEntropyLoss()

    def closure():

        optimizer.zero_grad()

        scaled_logits = (
            validation_logits
            / temperature
        )

        loss = criterion(
            scaled_logits,
            validation_labels,
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    return temperature.detach().item()


def main():

    print("=" * 70)
    print("Dermalens AI — Temperature Scaling")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    print("\nLoading datasets...")

    (
        _,
        validation_loader,
        test_loader,
    ) = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    print("\nLoading best Baseline A checkpoint...")

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

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print("\nCollecting validation logits...")

    validation_logits, validation_labels = (
        collect_logits(
            model,
            validation_loader,
        )
    )

    print("\nCollecting test logits...")

    test_logits, test_labels = (
        collect_logits(
            model,
            test_loader,
        )
    )

    print("\nFinding optimal temperature...")

    temperature = find_temperature(
        validation_logits,
        validation_labels,
    )

    print(
        f"Optimal temperature: "
        f"{temperature:.4f}"
    )

    # ---------------------------------------------------------
    # BEFORE CALIBRATION
    # ---------------------------------------------------------

    test_probabilities_before = torch.softmax(
        test_logits,
        dim=1,
    )

    predictions_before = torch.argmax(
        test_probabilities_before,
        dim=1,
    )

    accuracy_before = accuracy_score(
        test_labels.numpy(),
        predictions_before.numpy(),
    )

    ece_before = calculate_ece(
        test_probabilities_before,
        test_labels,
        NUM_BINS,
    )

    brier_before = calculate_brier_score(
        test_probabilities_before,
        test_labels,
        DermalensResNet50.NUM_CLASSES,
    )

    # ---------------------------------------------------------
    # AFTER CALIBRATION
    # ---------------------------------------------------------

    test_probabilities_after = torch.softmax(
        test_logits / temperature,
        dim=1,
    )

    predictions_after = torch.argmax(
        test_probabilities_after,
        dim=1,
    )

    accuracy_after = accuracy_score(
        test_labels.numpy(),
        predictions_after.numpy(),
    )

    ece_after = calculate_ece(
        test_probabilities_after,
        test_labels,
        NUM_BINS,
    )

    brier_after = calculate_brier_score(
        test_probabilities_after,
        test_labels,
        DermalensResNet50.NUM_CLASSES,
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALIBRATION COMPARISON")
    print("=" * 70)

    print(
        "\n                         Before       After"
    )

    print(
        f"Accuracy              "
        f"{accuracy_before:.4f}      "
        f"{accuracy_after:.4f}"
    )

    print(
        f"ECE                   "
        f"{ece_before:.4f}      "
        f"{ece_after:.4f}"
    )

    print(
        f"Brier Score           "
        f"{brier_before:.4f}      "
        f"{brier_after:.4f}"
    )

    print(
        "\nTemperature:"
        f" {temperature:.4f}"
    )

    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    Path("reports").mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        [
            {
                "method": "before_temperature_scaling",
                "temperature": 1.0,
                "accuracy": accuracy_before,
                "ece": ece_before,
                "brier_score": brier_before,
            },
            {
                "method": "after_temperature_scaling",
                "temperature": temperature,
                "accuracy": accuracy_after,
                "ece": ece_after,
                "brier_score": brier_after,
            },
        ]
    )

    results.to_csv(
        "reports/baseline_a_temperature_scaling.csv",
        index=False,
    )

    print(
        "Saved:"
        "\n- reports/baseline_a_temperature_scaling.csv"
    )

    print("\n" + "=" * 70)
    print("TEMPERATURE SCALING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()