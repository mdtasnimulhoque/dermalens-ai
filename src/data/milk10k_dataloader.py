
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.milk10k_dataset import MILK10kDataset
from src.data.paths import MILK10K_IMAGE_ROOT


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = MILK10K_IMAGE_ROOT
MASTER_INDEX = PROJECT_ROOT / "data" / "metadata" / "milk10k_master_index.csv"
SPLIT_FILE = PROJECT_ROOT / "data" / "metadata" / "milk10k_splits.csv"


# ---------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------

def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_dataset = MILK10kDataset(
        master_index_csv=str(MASTER_INDEX),
        split_csv=str(SPLIT_FILE),
        split="train",
        image_root=str(DATASET_ROOT),
    )

    validation_dataset = MILK10kDataset(
        master_index_csv=str(MASTER_INDEX),
        split_csv=str(SPLIT_FILE),
        split="validation",
        image_root=str(DATASET_ROOT),
    )

    test_dataset = MILK10kDataset(
        master_index_csv=str(MASTER_INDEX),
        split_csv=str(SPLIT_FILE),
        split="test",
        image_root=str(DATASET_ROOT),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, validation_loader, test_loader