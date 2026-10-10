
from pathlib import Path
import json

from PIL import Image

from src.data.paths import MILK10K_IMAGE_ROOT


# ============================================================
# Dermalens AI — MILK10k Image Integrity Checker
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = MILK10K_IMAGE_ROOT
EXPECTED_IMAGES = 10480

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "milk10k_image_integrity_report.json"


def check_images():
    print("=" * 60)
    print("Dermalens AI — MILK10k Image Integrity Check")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print("\nERROR: Dataset not found:")
        print(DATASET_DIR)
        return

    if not DATASET_DIR.is_dir():
        print("\nERROR: Dataset path is not a directory:")
        print(DATASET_DIR)
        return

    image_files = sorted(DATASET_DIR.rglob("*.jpg"))

    print(f"\nJPEG files found: {len(image_files)}")
    print(f"Expected images:  {EXPECTED_IMAGES}")

    if len(image_files) != EXPECTED_IMAGES:
        print("\nWARNING: Image count does not match expected count.")

    valid_images = 0
    invalid_images = 0
    suspicious_images = []
    dimensions = {}

    print("\nChecking image readability...")
    print("This may take a little while.")

    for index, image_path in enumerate(image_files, start=1):
        try:
            # Verify the file structure.
            with Image.open(image_path) as image:
                image.verify()

            # Reopen after verify() to read image properties.
            with Image.open(image_path) as image:
                width, height = image.size

            valid_images += 1

            dimension_key = f"{width}x{height}"
            dimensions[dimension_key] = (
                dimensions.get(dimension_key, 0) + 1
            )

            # Flag unusually small images.
            if width < 100 or height < 100:
                suspicious_images.append({
                    "file": str(image_path),
                    "width": width,
                    "height": height,
                    "reason": "Very small image",
                })

        except Exception as exc:
            invalid_images += 1

            suspicious_images.append({
                "file": str(image_path),
                "reason": f"Unreadable image: {exc}",
            })

        # Progress every 500 images.
        if index % 500 == 0 or index == len(image_files):
            print(
                f"Checked {index}/{len(image_files)} "
                f"| Valid: {valid_images} "
                f"| Invalid: {invalid_images}"
            )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {
        "dataset": "MILK10k",
        "expected_images": EXPECTED_IMAGES,
        "images_found": len(image_files),
        "valid_images": valid_images,
        "invalid_images": invalid_images,
        "suspicious_images": suspicious_images,
        "image_dimensions": dimensions,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 60)
    print("IMAGE INTEGRITY CHECK COMPLETE")
    print("=" * 60)

    print(f"\nImages found:       {len(image_files)}")
    print(f"Valid images:       {valid_images}")
    print(f"Invalid images:     {invalid_images}")
    print(f"Suspicious images:  {len(suspicious_images)}")

    print("\nReport saved to:")
    print(REPORT_PATH)

    if invalid_images == 0:
        print("\n✓ All images are readable.")
    else:
        print("\n⚠ Some images could not be read.")

    if not suspicious_images:
        print("✓ No suspiciously small images detected.")
    else:
        print("⚠ Suspicious images were detected.")


if __name__ == "__main__":
    check_images()