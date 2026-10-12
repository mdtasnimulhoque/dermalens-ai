
"""Unit tests for configurable local MILK10k paths."""

import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import src.data.paths as dataset_paths


class TestDatasetPaths(unittest.TestCase):
    """Test default paths and environment-variable overrides."""

    def tearDown(self):
        importlib.reload(dataset_paths)

    def test_default_paths_use_downloads_directory(self):
        with patch.dict(
            os.environ,
            {"HOME": "/tmp/dermalens-test-home"},
            clear=True,
        ):
            paths = importlib.reload(dataset_paths)

        self.assertEqual(
            paths.MILK10K_IMAGE_ROOT,
            Path("/tmp/dermalens-test-home/Downloads/MILK10k_Training_Input"),
        )
        self.assertEqual(
            paths.MILK10K_METADATA_ROOT,
            Path("/tmp/dermalens-test-home/Downloads"),
        )

    def test_environment_variables_override_defaults(self):
        environment = {
            "HOME": "/tmp/dermalens-test-home",
            "DERMALENS_MILK10K_IMAGE_ROOT": "/datasets/milk10k/images",
            "DERMALENS_MILK10K_METADATA_ROOT": "/datasets/milk10k/metadata",
        }
        with patch.dict(os.environ, environment, clear=True):
            paths = importlib.reload(dataset_paths)

        self.assertEqual(
            paths.MILK10K_IMAGE_ROOT,
            Path("/datasets/milk10k/images"),
        )
        self.assertEqual(
            paths.MILK10K_METADATA_ROOT,
            Path("/datasets/milk10k/metadata"),
        )

    def test_tilde_in_overrides_is_expanded(self):
        environment = {
            "HOME": "/tmp/dermalens-test-home",
            "DERMALENS_MILK10K_IMAGE_ROOT": "~/datasets/images",
            "DERMALENS_MILK10K_METADATA_ROOT": "~/datasets/metadata",
        }
        with patch.dict(os.environ, environment, clear=True):
            paths = importlib.reload(dataset_paths)

        self.assertEqual(
            paths.MILK10K_IMAGE_ROOT,
            Path("/tmp/dermalens-test-home/datasets/images"),
        )
        self.assertEqual(
            paths.MILK10K_METADATA_ROOT,
            Path("/tmp/dermalens-test-home/datasets/metadata"),
        )


if __name__ == "__main__":
    unittest.main()
