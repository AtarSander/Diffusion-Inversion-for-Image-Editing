"""Shared COCO caption dataset paths and naming helpers."""

from pathlib import Path

from diff_inversion.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

COCO_ANNOTATIONS_2014_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)
COCO_ANNOTATIONS_2017_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def get_coco_caption_source(year: str, split: str) -> tuple[str, str, str]:
    """Return the download URL, archive member path, and extracted JSON filename."""
    if year == "2014":
        return (
            COCO_ANNOTATIONS_2014_URL,
            f"annotations/captions_{split}2014.json",
            f"captions_{split}2014.json",
        )

    return (
        COCO_ANNOTATIONS_2017_URL,
        f"annotations/captions_{split}2017.json",
        f"captions_{split}2017.json",
    )


def get_raw_coco_dir() -> Path:
    """Return the root directory for raw COCO caption files."""
    return RAW_DATA_DIR / "coco"


def get_processed_coco_dir() -> Path:
    """Return the root directory for processed COCO prompt files."""
    return PROCESSED_DATA_DIR / "coco"
