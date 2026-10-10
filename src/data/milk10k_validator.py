
from pathlib import Path

import pandas as pd

from src.data.paths import MILK10K_IMAGE_ROOT, MILK10K_METADATA_ROOT


# ============================================================
# Dermalens AI — MILK10k Dataset Integrity Validator
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = MILK10K_IMAGE_ROOT

METADATA_FILE = (
    MILK10K_METADATA_ROOT / "MILK10k_Training_Metadata.csv"
)
SUPPLEMENT_FILE = (
    MILK10K_METADATA_ROOT / "MILK10k_Training_Supplement.csv"
)
GROUND_TRUTH_FILE = (
    MILK10K_METADATA_ROOT / "MILK10k_Training_GroundTruth.csv"
)

EXPECTED_LESIONS = 5240
EXPECTED_IMAGES = 10480

CLASSES = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]


def check_file_exists(path: Path) -> bool:
    """Check whether a required file or directory exists."""
    exists = path.exists()

    status = "OK" if exists else "MISSING"
    print(f"[{status}] {path}")

    return exists


def validate_metadata(metadata: pd.DataFrame) -> dict:
    """Validate MILK10k training metadata."""

    print("\n--- Metadata validation ---")

    result = {}

    result["rows"] = len(metadata)
    result["unique_lesions"] = metadata["lesion_id"].nunique()
    result["unique_images"] = metadata["isic_id"].nunique()

    print(f"Metadata rows:       {result['rows']}")
    print(f"Unique lesions:      {result['unique_lesions']}")
    print(f"Unique image IDs:    {result['unique_images']}")

    result["correct_row_count"] = result["rows"] == EXPECTED_IMAGES
    result["correct_lesion_count"] = (
        result["unique_lesions"] == EXPECTED_LESIONS
    )
    result["unique_image_ids"] = (
        result["unique_images"] == EXPECTED_IMAGES
    )

    # Every lesion should have exactly two images.
    images_per_lesion = metadata.groupby("lesion_id").size()

    result["lesions_with_two_images"] = int(
        (images_per_lesion == 2).sum()
    )

    result["all_lesions_have_two_images"] = (
        result["lesions_with_two_images"] == EXPECTED_LESIONS
    )

    print(
        f"Lesions with exactly 2 images: "
        f"{result['lesions_with_two_images']}/{EXPECTED_LESIONS}"
    )

    # Check modalities.
    modality_counts = (
        metadata.groupby(["lesion_id", "image_type"])
        .size()
        .unstack(fill_value=0)
    )

    required_modalities = {
        "clinical: close-up",
        "dermoscopic",
    }

    actual_modalities = set(metadata["image_type"].dropna().unique())

    result["required_modalities_present"] = (
        required_modalities.issubset(actual_modalities)
    )

    result["lesions_with_both_modalities"] = int(
        (
            (modality_counts.get("clinical: close-up", 0) == 1)
            & (modality_counts.get("dermoscopic", 0) == 1)
        ).sum()
    )

    result["all_lesions_have_both_modalities"] = (
        result["lesions_with_both_modalities"] == EXPECTED_LESIONS
    )

    print(
        f"Lesions with clinical + dermoscopic image: "
        f"{result['lesions_with_both_modalities']}/{EXPECTED_LESIONS}"
    )

    # Duplicate checks.
    duplicate_lesion_rows = int(
        metadata.duplicated(
            subset=["lesion_id", "image_type"]
        ).sum()
    )

    duplicate_image_ids = int(
        metadata["isic_id"].duplicated().sum()
    )

    result["duplicate_lesion_modality_rows"] = duplicate_lesion_rows
    result["duplicate_image_ids"] = duplicate_image_ids

    print(f"Duplicate lesion/modality rows: {duplicate_lesion_rows}")
    print(f"Duplicate ISIC image IDs:       {duplicate_image_ids}")

    return result


def validate_ground_truth(
    ground_truth: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict:
    """Validate MILK10k lesion-level ground truth."""

    print("\n--- Ground truth validation ---")

    result = {}

    result["rows"] = len(ground_truth)
    result["unique_lesions"] = ground_truth["lesion_id"].nunique()

    print(f"Ground truth rows:    {result['rows']}")
    print(f"Unique GT lesions:    {result['unique_lesions']}")

    result["correct_row_count"] = (
        result["rows"] == EXPECTED_LESIONS
    )

    result["unique_lesion_ids"] = (
        result["unique_lesions"] == EXPECTED_LESIONS
    )

    # Check class columns.
    missing_classes = [
        class_name
        for class_name in CLASSES
        if class_name not in ground_truth.columns
    ]

    result["missing_class_columns"] = missing_classes

    if missing_classes:
        print(f"Missing class columns: {missing_classes}")
        return result

    # Check that each lesion has exactly one class.
    label_values = ground_truth[CLASSES]

    invalid_values = ~label_values.isin([0.0, 1.0])

    result["invalid_label_values"] = int(invalid_values.sum().sum())

    label_sum = label_values.sum(axis=1)

    result["lesions_with_one_label"] = int(
        (label_sum == 1).sum()
    )

    result["all_lesions_have_one_label"] = (
        result["lesions_with_one_label"] == EXPECTED_LESIONS
    )

    print(
        f"Lesions with exactly one class: "
        f"{result['lesions_with_one_label']}/{EXPECTED_LESIONS}"
    )

    print(f"Invalid label values: {result['invalid_label_values']}")

    # Class distribution.
    class_counts = label_values.sum().to_dict()

    result["class_distribution"] = {
        key: int(value)
        for key, value in class_counts.items()
    }

    print("\nClass distribution:")

    for class_name, count in result["class_distribution"].items():
        print(f"  {class_name:8} {count}")

    # Compare lesion IDs.
    metadata_lesions = set(metadata["lesion_id"])
    ground_truth_lesions = set(ground_truth["lesion_id"])

    missing_in_ground_truth = metadata_lesions - ground_truth_lesions
    missing_in_metadata = ground_truth_lesions - metadata_lesions

    result["metadata_lesions_missing_from_ground_truth"] = len(
        missing_in_ground_truth
    )

    result["ground_truth_lesions_missing_from_metadata"] = len(
        missing_in_metadata
    )

    print(
        "Metadata lesions missing from GT: "
        f"{len(missing_in_ground_truth)}"
    )

    print(
        "GT lesions missing from metadata: "
        f"{len(missing_in_metadata)}"
    )

    return result


def validate_image_files(metadata: pd.DataFrame) -> dict:
    """Validate the actual image files on disk."""

    print("\n--- Image file validation ---")

    result = {}

    image_files = list(DATASET_DIR.rglob("*.jpg"))

    result["jpg_files"] = len(image_files)

    print(f"JPEG files found: {result['jpg_files']}")
    print(f"Expected JPEG files: {EXPECTED_IMAGES}")

    result["correct_image_count"] = (
        result["jpg_files"] == EXPECTED_IMAGES
    )

    # Build mapping from filename stem to file path.
    actual_images = {
        image_path.stem: image_path
        for image_path in image_files
    }

    metadata_image_ids = set(metadata["isic_id"])
    actual_image_ids = set(actual_images)

    missing_files = metadata_image_ids - actual_image_ids
    unexpected_files = actual_image_ids - metadata_image_ids

    result["metadata_images_missing_on_disk"] = len(missing_files)
    result["unexpected_images_on_disk"] = len(unexpected_files)

    print(f"Metadata images missing on disk: {len(missing_files)}")
    print(f"Unexpected images on disk: {len(unexpected_files)}")

    return result


def main():
    print("=" * 60)
    print("Dermalens AI — MILK10k Dataset Validator")
    print("=" * 60)

    print("\nChecking required files...")

    required_paths = [
        DATASET_DIR,
        METADATA_FILE,
        SUPPLEMENT_FILE,
        GROUND_TRUTH_FILE,
    ]

    all_exist = all(
        check_file_exists(path)
        for path in required_paths
    )

    if not all_exist:
        print("\nERROR: One or more required files are missing.")
        return

    print("\nLoading CSV files...")

    metadata = pd.read_csv(METADATA_FILE)
    supplement = pd.read_csv(SUPPLEMENT_FILE)
    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

    print("CSV files loaded successfully.")

    metadata_report = validate_metadata(metadata)

    print("\n--- Supplement validation ---")
    print(f"Supplement rows: {len(supplement)}")
    print(
        f"Unique supplement image IDs: "
        f"{supplement['isic_id'].nunique()}"
    )

    ground_truth_report = validate_ground_truth(
        ground_truth,
        metadata,
    )

    image_report = validate_image_files(metadata)

    # Final summary.
    report = {
        "dataset": "MILK10k",
        "expected_lesions": EXPECTED_LESIONS,
        "expected_images": EXPECTED_IMAGES,
        "metadata": metadata_report,
        "ground_truth": ground_truth_report,
        "images": image_report,
    }

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    print("\nThe dataset has NOT been moved or modified.")
    print("This script only inspected the dataset.")

    print("\nNext step:")
    print("Review the validation results before training.")


if __name__ == "__main__":
    main()