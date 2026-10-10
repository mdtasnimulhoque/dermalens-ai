from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.data.milk10k_dataloader import create_dataloaders
from src.models.resnet50_baseline import DermalensResNet50


CHECKPOINT = "models/checkpoints/baseline_a_best.pt"
OUTPUT_FILE = "reports/baseline_a_predictions.csv"
BATCH_SIZE = 32

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def main():

    print("=" * 70)
    print("Dermalens AI — Baseline A Error Analysis")
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

    class_names = DermalensResNet50.CLASS_NAMES

    records = []

    print("\nRunning predictions...")

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(DEVICE)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            confidences = torch.max(
                probabilities,
                dim=1,
            ).values

            for i in range(len(predictions)):

                true_index = batch["label"][i].item()
                predicted_index = predictions[i].item()

                records.append(
                    {
                        "lesion_id": batch["lesion_id"][i],
                        "image_id": batch["image_id"][i],
                        "true_class": class_names[true_index],
                        "predicted_class": class_names[predicted_index],
                        "confidence": confidences[i].item(),
                        "correct": (
                            true_index == predicted_index
                        ),
                    }
                )

    df = pd.DataFrame(records)

    Path("reports").mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved predictions to: {OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("ERROR SUMMARY")
    print("=" * 70)

    total = len(df)
    correct = df["correct"].sum()
    incorrect = total - correct

    print(f"Total test images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(
        f"Accuracy: {correct / total:.4f}"
    )

    print("\nMost common mistakes:")

    errors = df[
        df["correct"] == False
    ]

    error_pairs = (
        errors
        .groupby(
            ["true_class", "predicted_class"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    print(
        error_pairs.head(20).to_string(
            index=False
        )
    )

    print("\nAverage confidence:")

    confidence_summary = (
        df.groupby("correct")["confidence"]
        .mean()
    )

    print(
        confidence_summary.to_string()
    )

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()