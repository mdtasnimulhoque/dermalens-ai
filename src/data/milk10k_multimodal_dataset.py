from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MILK10kMultimodalDataset(Dataset):

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
        name: index
        for index, name in enumerate(CLASS_NAMES)
    }

    SEX_MAPPING = {
        "female": 0,
        "male": 1,
        "unknown": 2,
    }

    SITE_MAPPING = {
        "trunk": 0,
        "head_neck_face": 1,
        "lower_extremity": 2,
        "upper_extremity": 3,
        "hand": 4,
        "foot": 5,
        "genital": 6,
        "unknown": 7,
    }

    NUM_SEX_CATEGORIES = 3
    NUM_SKIN_TONE_CATEGORIES = 6
    NUM_SITE_CATEGORIES = 8

    # 1 age + 3 sex + 6 skin tone + 8 site = 18
    NUM_METADATA_FEATURES = 18

    def __init__(
        self,
        master_index_csv,
        split_csv,
        split,
        image_root,
        transform=None,
    ):
        self.image_root = Path(image_root)

        master = pd.read_csv(master_index_csv)
        splits = pd.read_csv(split_csv)

        required_columns = [
            "lesion_id",
            "image_type",
            "isic_id",
            "image_path",
            "diagnosis",
            "age_approx",
            "sex",
            "skin_tone_class",
            "site",
        ]

        missing = [
            column
            for column in required_columns
            if column not in master.columns
        ]

        if missing:
            raise ValueError(
                "Master index is missing required columns: "
                + ", ".join(missing)
            )

        if "split" not in splits.columns:
            raise ValueError(
                "Split CSV does not contain a 'split' column."
            )

        split_data = (
            splits[["lesion_id", "split"]]
            .drop_duplicates(subset=["lesion_id"])
        )

        df = master.merge(
            split_data,
            on="lesion_id",
            how="inner",
        )

        df = df[df["split"] == split].copy()

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

    def _encode_age(self, value):

        if pd.isna(value):
            return 0.0

        try:
            age = float(value)
        except (TypeError, ValueError):
            return 0.0

        return age / 100.0

    def _encode_sex(self, value):

        if pd.isna(value):
            return 2

        value = str(value).strip().lower()

        return self.SEX_MAPPING.get(
            value,
            2,
        )

    def _encode_skin_tone(self, value):

        if pd.isna(value):
            return 0

        try:
            skin_tone = int(float(value))
        except (TypeError, ValueError):
            return 0

        if not 0 <= skin_tone <= 5:
            return 0

        return skin_tone

    def _encode_site(self, value):

        if pd.isna(value):
            return 7

        value = str(value).strip().lower()

        return self.SITE_MAPPING.get(
            value,
            7,
        )

    def _build_metadata_vector(self, row):

        age = self._encode_age(
            row["age_approx"]
        )

        sex = self._encode_sex(
            row["sex"]
        )

        skin_tone = self._encode_skin_tone(
            row["skin_tone_class"]
        )

        site = self._encode_site(
            row["site"]
        )

        sex_one_hot = torch.zeros(
            3,
            dtype=torch.float32,
        )
        sex_one_hot[sex] = 1.0

        skin_tone_one_hot = torch.zeros(
            6,
            dtype=torch.float32,
        )
        skin_tone_one_hot[skin_tone] = 1.0

        site_one_hot = torch.zeros(
            8,
            dtype=torch.float32,
        )
        site_one_hot[site] = 1.0

        metadata = torch.cat(
            [
                torch.tensor(
                    [age],
                    dtype=torch.float32,
                ),
                sex_one_hot,
                skin_tone_one_hot,
                site_one_hot,
            ]
        )

        return metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = (
            self.image_root
            / row["image_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        metadata = self._build_metadata_vector(
            row
        )

        label_name = row["diagnosis"]

        if label_name not in self.CLASS_TO_INDEX:
            raise ValueError(
                f"Unknown diagnosis: {label_name}"
            )

        label = self.CLASS_TO_INDEX[
            label_name
        ]

        return {
            "image": image,
            "metadata": metadata,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "lesion_id": row["lesion_id"],
            "image_id": row["isic_id"],
            "diagnosis": label_name,
        }
