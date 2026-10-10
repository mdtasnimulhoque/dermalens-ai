
from pathlib import Path
import json

import pandas as pd
from PIL import Image

from src.data.paths import MILK10K_IMAGE_ROOT


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

IMAGE_ROOT = MILK10K_IMAGE_ROOT

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "milk10k_dataset_audit.json"


# =========================================================
# Expected dataset structure
# =========================================================

EXPECTED_IMAGES = 10_480
EXPECTED_LESIONS = 5_240

EXPECTED_IMAGE_TYPES = {
    "clinical: close-up",
    "dermoscopic",
}

EXPECTED_DIAGNOSES = {
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
}


# =========================================================
# Helpers
# =========================================================

def resolve_image_path(path_string: str) -> Path:
    """Resolve an index image path against the configured image root."""
    path = Path(path_string).expanduser()

    if path.is_absolute():
        return path

    return IMAGE_ROOT / path


def check_required_paths() -> None:
    """Check that the index and image directory exist."""
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(
            f"Master index not found:\n{INDEX_PATH}"
        )

    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_ROOT}"
        )


def record_check(report, name, passed, details=None):
    """Record an individual audit check."""
    report["checks"][name] = {
        "passed": bool(passed),
        "details": details,
    }


# =========================================================
# Main audit
# =========================================================

def main() -> None:
    print("=" * 72)
    print("Dermalens AI — MILK10k Dataset Audit")
    print("=" * 72)

    check_required_paths()

    print("\nLoading master index...")
    df = pd.read_csv(INDEX_PATH)

    report = {
        "dataset": "MILK10k",
        "index_path": str(INDEX_PATH),
        "image_root": str(IMAGE_ROOT),
        "checks": {},
        "summary": {},
        "limitations": [],
    }

    # =====================================================
    # Basic counts
    # =====================================================

    print("\n[1/12] Checking dataset counts...")

    image_count = len(df)
    lesion_count = df["lesion_id"].nunique()

    record_check(
        report,
        "expected_image_count",
        image_count == EXPECTED_IMAGES,
        {
            "expected": EXPECTED_IMAGES,
            "actual": image_count,
        },
    )

    record_check(
        report,
        "expected_lesion_count",
        lesion_count == EXPECTED_LESIONS,
        {
            "expected": EXPECTED_LESIONS,
            "actual": lesion_count,
        },
    )

    print(f"Images:  {image_count:,}")
    print(f"Lesions: {lesion_count:,}")

    # =====================================================
    # Required columns
    # =====================================================

    print("\n[2/12] Checking required columns...")

    required_columns = {
        "lesion_id",
        "isic_id",
        "image_type",
        "image_path",
        "diagnosis",
        "age_approx",
        "sex",
        "skin_tone_class",
        "site",
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    record_check(
        report,
        "required_columns",
        len(missing_columns) == 0,
        {"missing_columns": missing_columns},
    )

    # Stop if essential columns are absent. This prevents
    # later checks from raising confusing KeyError exceptions.
    essential_columns = {
        "lesion_id",
        "isic_id",
        "image_type",
        "image_path",
        "diagnosis",
    }

    missing_essential = essential_columns - set(df.columns)

    if missing_essential:
        report["summary"]["audit_passed"] = False
        report["summary"]["total_checks"] = len(report["checks"])
        report["summary"]["passed_checks"] = sum(
            item["passed"] for item in report["checks"].values()
        )
        report["summary"]["failed_checks"] = (
            report["summary"]["total_checks"]
            - report["summary"]["passed_checks"]
        )

        print(
            "\nAudit cannot continue. Essential columns are missing:",
            sorted(missing_essential),
        )
        return

    # =====================================================
    # Image IDs
    # =====================================================

    print("\n[3/12] Checking image IDs...")

    duplicate_image_ids = int(
        df["isic_id"].duplicated().sum()
    )

    record_check(
        report,
        "unique_image_ids",
        duplicate_image_ids == 0,
        {"duplicate_count": duplicate_image_ids},
    )

    # =====================================================
    # Lesion/image relationship
    # =====================================================

    print("\n[4/12] Checking lesion/image relationships...")

    images_per_lesion = df.groupby("lesion_id").size()
    bad_lesions = images_per_lesion[images_per_lesion != 2]

    record_check(
        report,
        "two_images_per_lesion",
        len(bad_lesions) == 0,
        {
            "bad_lesion_count": int(len(bad_lesions)),
            "examples": bad_lesions.head(10).to_dict(),
        },
    )

    # =====================================================
    # Image modality pairing
    # =====================================================

    print("\n[5/12] Checking clinical/dermoscopic pairing...")

    modality_counts = (
        df.groupby("lesion_id")["image_type"].nunique()
    )

    bad_modality_count = int(
        (modality_counts != 2).sum()
    )

    expected_modalities = (
        df.groupby("lesion_id")["image_type"].apply(set)
    )

    bad_modalities = expected_modalities[
        ~expected_modalities.apply(
            lambda values: values == EXPECTED_IMAGE_TYPES
        )
    ]

    record_check(
        report,
        "clinical_dermoscopic_pairing",
        bad_modality_count == 0 and len(bad_modalities) == 0,
        {
            "bad_lesion_count": int(len(bad_modalities)),
            "examples": {
                str(key): sorted(list(values))
                for key, values in bad_modalities.head(10).items()
            },
        },
    )

    # =====================================================
    # Duplicate lesion/modality rows
    # =====================================================

    print("\n[6/12] Checking duplicate lesion/modality rows...")

    duplicate_pairs = int(
        df.duplicated(
            subset=["lesion_id", "image_type"]
        ).sum()
    )

    record_check(
        report,
        "unique_lesion_modality_pairs",
        duplicate_pairs == 0,
        {"duplicate_count": duplicate_pairs},
    )

    # =====================================================
    # Diagnosis validation
    # =====================================================

    print("\n[7/12] Checking diagnoses...")

    missing_diagnosis = int(df["diagnosis"].isna().sum())

    unexpected_diagnoses = sorted(
        set(df["diagnosis"].dropna()) - EXPECTED_DIAGNOSES
    )

    record_check(
        report,
        "valid_diagnoses",
        missing_diagnosis == 0 and len(unexpected_diagnoses) == 0,
        {
            "missing_count": missing_diagnosis,
            "unexpected_diagnoses": unexpected_diagnoses,
        },
    )

    diagnosis_per_lesion = (
        df.groupby("lesion_id")["diagnosis"].nunique()
    )

    inconsistent_diagnosis = diagnosis_per_lesion[
        diagnosis_per_lesion != 1
    ]

    record_check(
        report,
        "consistent_diagnosis_per_lesion",
        len(inconsistent_diagnosis) == 0,
        {
            "inconsistent_lesion_count": int(
                len(inconsistent_diagnosis)
            ),
        },
    )

    # =====================================================
    # Missing metadata
    # =====================================================

    print("\n[8/12] Checking metadata completeness...")

    metadata_columns = [
        "age_approx",
        "sex",
        "skin_tone_class",
        "site",
    ]

    missing_metadata = {
        column: (
            int(df[column].isna().sum())
            if column in df.columns
            else None
        )
        for column in metadata_columns
    }

    report["summary"]["missing_metadata"] = missing_metadata

    print("Missing metadata:")
    for column, count in missing_metadata.items():
        value = "column missing" if count is None else f"{count:,}"
        print(f"  {column}: {value}")

    # =====================================================
    # Image paths
    # =====================================================

    print("\n[9/12] Checking image paths...")

    resolved_paths = [
        resolve_image_path(path_string)
        for path_string in df["image_path"]
    ]

    missing_paths = [
        str(path)
        for path in resolved_paths
        if not path.is_file()
    ]

    record_check(
        report,
        "image_paths_exist",
        len(missing_paths) == 0,
        {
            "missing_path_count": len(missing_paths),
            "examples": missing_paths[:10],
        },
    )

    print(f"Missing image files: {len(missing_paths):,}")

    # =====================================================
    # Image integrity + dimensions
    # =====================================================

    print("\n[10/12] Checking image integrity and dimensions...")

    invalid_images = []
    suspicious_images = []

    min_width = None
    min_height = None
    max_width = None
    max_height = None

    for i, path in enumerate(resolved_paths, start=1):
        # Don't attempt to open files that are already missing.
        if not path.is_file():
            continue

        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                width, height = image.size

            if min_width is None:
                min_width = width
                min_height = height
                max_width = width
                max_height = height
            else:
                min_width = min(min_width, width)
                min_height = min(min_height, height)
                max_width = max(max_width, width)
                max_height = max(max_height, height)

            if width < 100 or height < 100:
                suspicious_images.append({
                    "path": str(path),
                    "width": width,
                    "height": height,
                })

        except Exception as exc:
            invalid_images.append({
                "path": str(path),
                "error": str(exc),
            })

        if i % 1000 == 0:
            print(f"  Checked {i:,}/{len(df):,}")

    record_check(
        report,
        "image_integrity",
        len(invalid_images) == 0,
        {
            "invalid_count": len(invalid_images),
            "examples": invalid_images[:10],
        },
    )

    record_check(
        report,
        "image_dimensions",
        len(suspicious_images) == 0,
        {
            "suspicious_count": len(suspicious_images),
            "examples": suspicious_images[:10],
        },
    )

    report["summary"]["image_dimensions"] = {
        "minimum_width": min_width,
        "minimum_height": min_height,
        "maximum_width": max_width,
        "maximum_height": max_height,
    }

    print(f"Invalid images:     {len(invalid_images):,}")
    print(f"Suspicious images:  {len(suspicious_images):,}")

    # =====================================================
    # Class distribution
    # =====================================================

    print("\n[11/12] Calculating lesion-level class distribution...")

    lesion_labels = df[["lesion_id", "diagnosis"]].drop_duplicates()

    class_counts = (
        lesion_labels["diagnosis"].value_counts().sort_index()
    )

    total_labelled_lesions = len(lesion_labels)

    if total_labelled_lesions:
        class_percentages = (
            class_counts / total_labelled_lesions * 100
        ).round(2)
    else:
        class_percentages = class_counts.astype(float)

    class_distribution = {}

    for diagnosis in class_counts.index:
        class_distribution[diagnosis] = {
            "lesions": int(class_counts[diagnosis]),
            "percentage": float(class_percentages[diagnosis]),
        }

    report["summary"]["class_distribution"] = class_distribution

    print("\nLesion-level class distribution:")

    for diagnosis, values in class_distribution.items():
        print(
            f"  {diagnosis:8s}"
            f" {values['lesions']:5d}"
            f" ({values['percentage']:6.2f}%)"
        )

    # =====================================================
    # Patient ID limitation
    # =====================================================

    print("\n[12/12] Checking patient identifier availability...")

    patient_columns = [
        column
        for column in df.columns
        if "patient" in column.lower()
    ]

    report["summary"]["patient_identifier_columns"] = patient_columns

    if patient_columns:
        print("Patient identifier columns found:", patient_columns)
    else:
        print("No explicit patient identifier found.")

        report["limitations"].append(
            "The current MILK10k training metadata does not "
            "contain an explicit patient identifier. Patient-level "
            "leakage cannot therefore be directly assessed from "
            "the available fields."
        )

    # =====================================================
    # Overall result and report
    # =====================================================

    passed_checks = [
        result["passed"]
        for result in report["checks"].values()
    ]

    report["summary"]["total_checks"] = len(passed_checks)
    report["summary"]["passed_checks"] = sum(passed_checks)
    report["summary"]["failed_checks"] = (
        len(passed_checks) - sum(passed_checks)
    )
    report["summary"]["audit_passed"] = all(passed_checks)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # =====================================================
    # Final output
    # =====================================================

    print("\n" + "=" * 72)

    if report["summary"]["audit_passed"]:
        print("DATASET AUDIT PASSED")
    else:
        print("DATASET AUDIT FOUND ISSUES")

    print("=" * 72)

    print(
        f"\nChecks passed: "
        f"{report['summary']['passed_checks']}/"
        f"{report['summary']['total_checks']}"
    )

    print(f"Audit report:\n{REPORT_PATH}")

    if report["limitations"]:
        print("\nImportant limitation:")
        for limitation in report["limitations"]:
            print(f"- {limitation}")


if __name__ == "__main__":
    main()