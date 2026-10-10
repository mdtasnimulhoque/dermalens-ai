from pathlib import Path

import pandas as pd


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "milk10k_master_index.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "milk10k_splits.csv"
)


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 42

TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("=" * 72)
    print("Dermalens AI — MILK10k Lesion-Safe Split")
    print("=" * 72)

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Master index not found:\n{INDEX_PATH}"
        )

    df = pd.read_csv(INDEX_PATH)

    print(f"\nLoaded images:  {len(df):,}")
    print(f"Loaded lesions: {df['lesion_id'].nunique():,}")

    # -----------------------------------------------------
    # Build one row per lesion
    # -----------------------------------------------------

    lesion_df = (
        df[
            ["lesion_id", "diagnosis"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(lesion_df) != 5_240:
        raise ValueError(
            f"Expected 5,240 unique lesions, "
            f"found {len(lesion_df):,}."
        )

    # -----------------------------------------------------
    # Stratified lesion-level split
    # -----------------------------------------------------

    train_lesions = []
    validation_lesions = []
    test_lesions = []

    print("\nCreating stratified lesion-level split...")

    for diagnosis, group in lesion_df.groupby(
        "diagnosis",
        sort=True
    ):

        # Deterministic shuffle for this class
        group = group.sample(
            frac=1,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

        n = len(group)

        train_count = int(
            n * TRAIN_FRACTION
        )

        validation_count = int(
            n * VALIDATION_FRACTION
        )

        # Everything remaining goes to test
        test_start = (
            train_count + validation_count
        )

        train_part = group.iloc[
            :train_count
        ]

        validation_part = group.iloc[
            train_count:test_start
        ]

        test_part = group.iloc[
            test_start:
        ]

        train_lesions.extend(
            train_part["lesion_id"].tolist()
        )

        validation_lesions.extend(
            validation_part["lesion_id"].tolist()
        )

        test_lesions.extend(
            test_part["lesion_id"].tolist()
        )

    # -----------------------------------------------------
    # Create split mapping
    # -----------------------------------------------------

    split_map = {}

    for lesion_id in train_lesions:
        split_map[lesion_id] = "train"

    for lesion_id in validation_lesions:
        split_map[lesion_id] = "validation"

    for lesion_id in test_lesions:
        split_map[lesion_id] = "test"

    if len(split_map) != 5_240:
        raise ValueError(
            "Some lesions were not assigned exactly once."
        )

    # -----------------------------------------------------
    # Apply split to image-level index
    # -----------------------------------------------------

    df["split"] = df["lesion_id"].map(
        split_map
    )

    if df["split"].isna().any():
        raise ValueError(
            "Some images have no split assignment."
        )

    # -----------------------------------------------------
    # Verify lesion leakage
    # -----------------------------------------------------

    print("\nChecking for lesion leakage...")

    lesions_per_split = {
        split: set(
            df.loc[
                df["split"] == split,
                "lesion_id",
            ]
        )
        for split in [
            "train",
            "validation",
            "test",
        ]
    }

    train_validation_overlap = (
        lesions_per_split["train"]
        & lesions_per_split["validation"]
    )

    train_test_overlap = (
        lesions_per_split["train"]
        & lesions_per_split["test"]
    )

    validation_test_overlap = (
        lesions_per_split["validation"]
        & lesions_per_split["test"]
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):
        raise ValueError(
            "LESION LEAKAGE DETECTED."
        )

    print("✓ No lesion overlap between splits.")

    # -----------------------------------------------------
    # Verify exactly two images per lesion
    # -----------------------------------------------------

    images_per_lesion = (
        df.groupby("lesion_id")
        .size()
    )

    if not (images_per_lesion == 2).all():
        raise ValueError(
            "Some lesions do not have exactly two images."
        )

    # -----------------------------------------------------
    # Verify every lesion has only one split
    # -----------------------------------------------------

    split_counts = (
        df.groupby("lesion_id")["split"]
        .nunique()
    )

    if not (split_counts == 1).all():
        raise ValueError(
            "Some lesions appear in multiple splits."
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "-" * 72)
    print("SPLIT SUMMARY")
    print("-" * 72)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = df[df["split"] == split]

        lesion_count = subset["lesion_id"].nunique()
        image_count = len(subset)

        print(
            f"{split:12s}"
            f" lesions: {lesion_count:5d}"
            f" | images: {image_count:5d}"
        )

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    print("\n" + "-" * 72)
    print("LESION-LEVEL CLASS DISTRIBUTION")
    print("-" * 72)

    lesion_split = (
        df[
            [
                "lesion_id",
                "diagnosis",
                "split",
            ]
        ]
        .drop_duplicates()
    )

    distribution = (
        lesion_split
        .groupby(
            ["split", "diagnosis"]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(distribution)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 72)
    print("LESION-SAFE SPLIT CREATED SUCCESSFULLY")
    print("=" * 72)

    print(f"\nOutput:")
    print(OUTPUT_PATH)

    print("\nRandom seed:")
    print(RANDOM_SEED)

    print("\nImportant:")
    print(
        "Both clinical and dermoscopic images from each "
        "lesion remain in the same split."
    )


if __name__ == "__main__":
    main()