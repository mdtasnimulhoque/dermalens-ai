
"""Portable local paths for the MILK10k research dataset."""

import os
from pathlib import Path

# Override these paths with environment variables on another machine.
MILK10K_IMAGE_ROOT = Path(
    os.environ.get(
        "DERMALENS_MILK10K_IMAGE_ROOT",
        str(Path.home() / "Downloads" / "MILK10k_Training_Input"),
    )
).expanduser()

MILK10K_METADATA_ROOT = Path(
    os.environ.get(
        "DERMALENS_MILK10K_METADATA_ROOT",
        str(Path.home() / "Downloads"),
    )
).expanduser()