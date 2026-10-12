
"""Tests for lesion-level dataset split integrity."""

import unittest

from src.data.split_integrity import find_split_integrity_errors


class TestSplitIntegrity(unittest.TestCase):
    def test_valid_lesions_have_no_errors(self):
        rows = [
            {"lesion_id": "L1", "split": "train"},
            {"lesion_id": "L1", "split": "train"},
            {"lesion_id": "L2", "split": "validation"},
            {"lesion_id": "L3", "split": "test"},
        ]

        self.assertEqual(find_split_integrity_errors(rows), [])

    def test_detects_lesion_in_multiple_splits(self):
        rows = [
            {"lesion_id": "L1", "split": "train"},
            {"lesion_id": "L1", "split": "test"},
        ]

        errors = find_split_integrity_errors(rows)

        self.assertTrue(any("multiple splits" in error for error in errors))

    def test_detects_missing_lesion_id(self):
        rows = [{"lesion_id": "", "split": "train"}]

        errors = find_split_integrity_errors(rows)

        self.assertTrue(any("missing lesion_id" in error for error in errors))

    def test_detects_invalid_split_name(self):
        rows = [{"lesion_id": "L1", "split": "testing"}]

        errors = find_split_integrity_errors(rows)

        self.assertTrue(any("invalid split" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
