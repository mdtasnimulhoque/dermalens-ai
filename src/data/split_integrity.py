
"""Validation helpers for lesion-level dataset splits."""

from collections import defaultdict

EXPECTED_SPLITS = {"train", "validation", "test"}


def find_split_integrity_errors(rows):
    """Return validation errors for rows containing lesion_id and split.

    Each lesion must have a non-empty ID and belong to exactly one
    recognized split. Different lesions may share a split.
    """
    errors = []
    lesion_splits = defaultdict(set)

    for row_number, row in enumerate(rows, start=1):
        lesion_id = str(row.get("lesion_id", "")).strip()
        split = str(row.get("split", "")).strip().lower()

        if not lesion_id or lesion_id.lower() == "none":
            errors.append(f"Row {row_number}: missing lesion_id")
            continue

        if not split or split not in EXPECTED_SPLITS:
            errors.append(
                f"Row {row_number}: invalid split for lesion {lesion_id!r}"
            )
            continue

        lesion_splits[lesion_id].add(split)

    for lesion_id, splits in sorted(lesion_splits.items()):
        if len(splits) > 1:
            errors.append(
                f"Lesion {lesion_id!r} appears in multiple splits: "
                f"{', '.join(sorted(splits))}"
            )

    return errors
