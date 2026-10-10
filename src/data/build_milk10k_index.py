
from pathlib import Path

import pandas as pd

from src.data.paths import MILK10K_IMAGE_ROOT, MILK10K_METADATA_ROOT


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = MILK10K_METADATA_ROOT / "MILK10k_Training_Metadata.csv"
GROUND_TRUTH_PATH = MILK10K_METADATA_ROOT / "MILK10k_Training_GroundTruth.csv"
SUPPLEMENT_PATH = MILK10K_METADATA_ROOT / "MILK10k_Training_Supplement.csv"

IMAGE_ROOT = MILK10K_IMAGE_ROOT

OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "milk10k_master_index.csv"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GROUND_TRUTH_CLASSES = [
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

MONET_COLUMNS = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def check_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def get_diagnosis(row: pd.Series) -> str:
    active_classes = [
        class_name
        for class_name in GROUND_TRUTH_CLASSES
        if row[class_name] == 1
    ]

    if len(active_classes) != 1:
        raise ValueError(
            f"Expected exactly one diagnosis for lesion "
            f"{row['lesion_id']}, found: {active_classes}"
        )

    return active_classes[0]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Dermalens AI — MILK10k Master Index Builder")
    print("=" * 70)

    # Check required files
    check_file(METADATA_PATH, "Training metadata")
    check_file(GROUND_TRUTH_PATH, "Training ground truth")
    check_file(SUPPLEMENT_PATH, "Training supplement")
    check_file(IMAGE_ROOT, "MILK10k image directory")

    print("\nLoading official CSV files...")

    metadata = pd.read_csv(METADATA_PATH)
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
    supplement = pd.read_csv(SUPPLEMENT_PATH)

    print(f"Metadata rows:      {len(metadata):,}")
    print(f"Ground truth rows:  {len(ground_truth):,}")
    print(f"Supplement rows:    {len(supplement):,}")

    # -----------------------------------------------------
    # Validate basic identifiers
    # -----------------------------------------------------

    required_metadata_columns = [
        "lesion_id",
        "image_type",
        "isic_id",
        "age_approx",
        "sex",
        "skin_tone_class",
        "site",
        *MONET_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    missing_ground_truth_columns = [
        column
        for column in ["lesion_id", *GROUND_TRUTH_CLASSES]
        if column not in ground_truth.columns
    ]

    if missing_ground_truth_columns:
        raise ValueError(
            f"Missing ground-truth columns: "
            f"{missing_ground_truth_columns}"
        )

    # -----------------------------------------------------
    # Validate uniqueness
    # -----------------------------------------------------

    if metadata["isic_id"].duplicated().any():
        raise ValueError(
            "Duplicate ISIC image IDs found in metadata."
        )

    if metadata["lesion_id"].duplicated().sum() != 5240:
        raise ValueError(
            "Unexpected lesion/image relationship in metadata."
        )

    if ground_truth["lesion_id"].duplicated().any():
        raise ValueError(
            "Duplicate lesion IDs found in ground truth."
        )

    # -----------------------------------------------------
    # Convert ground truth one-hot labels into diagnosis
    # -----------------------------------------------------

    print("\nCreating diagnosis labels...")

    ground_truth = ground_truth.copy()
    ground_truth["diagnosis"] = ground_truth.apply(
        get_diagnosis,
        axis=1,
    )

    diagnosis_map = ground_truth[
        ["lesion_id", "diagnosis"]
    ]

    # -----------------------------------------------------
    # Merge metadata + ground truth
    # -----------------------------------------------------

    print("Merging metadata with ground truth...")

    index = metadata.merge(
        diagnosis_map,
        on="lesion_id",
        how="left",
        validate="many_to_one",
    )

    if index["diagnosis"].isna().any():
        missing = index.loc[
            index["diagnosis"].isna(),
            "lesion_id",
        ].unique()

        raise ValueError(
            f"Missing diagnosis for lesions: {missing[:10]}"
        )

    # -----------------------------------------------------
    # Add image paths
    # -----------------------------------------------------

    print("Resolving image paths...")

    def resolve_image_path(row: pd.Series) -> str:
        lesion_directory = IMAGE_ROOT / row["lesion_id"]

        image_path = lesion_directory / f"{row['isic_id']}.jpg"

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Could not locate image:\n"
                f"Lesion: {row['lesion_id']}\n"
                f"ISIC ID: {row['isic_id']}\n"
                f"Expected: {image_path}"
            )

        return str(image_path)

    index["image_path"] = index.apply(
        resolve_image_path,
        axis=1,
    )

    # -----------------------------------------------------
    # Select research columns
    # -----------------------------------------------------

    output_columns = [
        "lesion_id",
        "isic_id",
        "image_type",
        "image_path",
        "diagnosis",
        "age_approx",
        "sex",
        "skin_tone_class",
        "site",
        "MONET_ulceration_crust",
        "MONET_hair",
        "MONET_vasculature_vessels",
        "MONET_erythema",
        "MONET_pigmented",
        "MONET_gel_water_drop_fluid_dermoscopy_liquid",
        "MONET_skin_markings_pen_ink_purple_pen",
    ]

    index = index[output_columns]

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    print("\nRunning final index checks...")

    expected_rows = 10_480

    if len(index) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"found {len(index):,}."
        )

    if index["isic_id"].nunique() != 10_480:
        raise ValueError(
            "Unexpected number of unique image IDs."
        )

    if index["lesion_id"].nunique() != 5_240:
        raise ValueError(
            "Unexpected number of unique lesions."
        )

    if index["image_path"].isna().any():
        raise ValueError("Some images have missing paths.")

    # Each lesion should have exactly two images
    images_per_lesion = index.groupby("lesion_id").size()

    if not (images_per_lesion == 2).all():
        raise ValueError(
            "Every lesion should have exactly two images."
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("MASTER INDEX CREATED SUCCESSFULLY")
    print("=" * 70)

    print("\nOutput:")
    print(OUTPUT_PATH)

    print(f"\nRows:             {len(index):,}")
    print(f"Unique lesions:   {index['lesion_id'].nunique():,}")
    print(f"Unique images:    {index['isic_id'].nunique():,}")

    print("\nImage types:")
    print(index["image_type"].value_counts())

    print("\nDiagnosis distribution:")
    print(index["diagnosis"].value_counts())

    print("\nNo image files were copied or moved.")
    print("The index points to the original dataset.")


if __name__ == "__main__":
    main()