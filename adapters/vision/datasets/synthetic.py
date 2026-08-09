from typing import Any
import torch
from core.base.dataset import BaseDataset
from core.base.io_types import IMAGE_TO_CATEGORY
from core.registry import DATASETS


@DATASETS.register("synthetic_image")
class SyntheticImageDataset(BaseDataset):

    io_type = IMAGE_TO_CATEGORY

    def __init__(self, params: dict[str, Any]) -> None:
        self._size: int = params.get("size", 100)
        self._num_classes: int = params.get("num_classes", 10)
        self._image_size: int = params.get("image_size", 64)
        self._channels: int = params.get("channels", 3)
        generator = torch.Generator().manual_seed(params.get("seed", 42))
        self._images = torch.randn(self._size, self._channels, self._image_size, self._image_size, generator=generator)
        self._labels = torch.randint(0, self._num_classes, (self._size,), generator=generator)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "pixel_values": self._images[idx],
            "labels": self._labels[idx],
        }

    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "pixel_values": torch.stack([s["pixel_values"] for s in samples]),
            "labels": torch.stack([s["labels"] for s in samples]),
        }
