from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MILK10kDataset(Dataset):
    """
    PyTorch dataset for MILK10k.

    Baseline A:
    Clinical close-up images only.

    The master index contains image paths and diagnoses.
    The split file determines train/validation/test membership.

    The split file contains two rows per lesion because each lesion
    has one clinical image and one dermoscopic image. We therefore
    reduce it to one split assignment per lesion before merging.
    """

    CLASS_NAMES = [
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

    CLASS_TO_INDEX = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }

    def __init__(
        self,
        master_index_csv: str,
        split_csv: str,
        split: str,
        image_root: str,
        transform=None,
    ):
        self.image_root = Path(image_root)

        master = pd.read_csv(master_index_csv)
        splits = pd.read_csv(split_csv)

        if "split" not in splits.columns:
            raise ValueError(
                "Split CSV does not contain a 'split' column."
            )

        # The split file contains two rows per lesion.
        # Keep exactly one split assignment per lesion.
        split_data = (
            splits[["lesion_id", "split"]]
            .drop_duplicates(subset=["lesion_id"])
        )

        # Merge lesion-level split assignments with the image-level index.
        df = master.merge(
            split_data,
            on="lesion_id",
            how="inner",
        )

        # Keep only the requested split.
        df = df[df["split"] == split].copy()

        # Baseline A uses clinical close-up images only.
        df = df[
            df["image_type"] == "clinical: close-up"
        ].copy()

        if df.empty:
            raise ValueError(
                f"No clinical images found for split: {split}"
            )

        self.data = df.reset_index(drop=True)

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = self.image_root / row["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_name = row["diagnosis"]
        label = self.CLASS_TO_INDEX[label_name]

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "lesion_id": row["lesion_id"],
            "image_id": row["isic_id"],
            "diagnosis": label_name,
        }