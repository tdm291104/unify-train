from typing import Any
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from core.base.model import BaseModel
from core.base.io_types import IMAGE_TO_CATEGORY
from core.registry import MODELS


@MODELS.register("resnet18")
class ResNet18Model(BaseModel):

    io_type = IMAGE_TO_CATEGORY

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._num_classes: int = 0

    def build(self, config: dict[str, Any]) -> None:
        self._num_classes = config.get("num_classes", 10)
        self._model = resnet18(weights=None)
        in_features = self._model.fc.in_features
        self._model.fc = torch.nn.Linear(in_features, self._num_classes)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        logits = self._model(batch["pixel_values"])
        result: dict[str, Any] = {"logits": logits}
        if "labels" in batch:
            result["loss"] = F.cross_entropy(logits, batch["labels"])
        return result

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self._model.state_dict(), f"{path}/resnet18.pt")

    def load(self, path: str) -> None:
        state = torch.load(f"{path}/resnet18.pt", map_location="cpu")
        self._model.load_state_dict(state)

    @property
    def raw_model(self) -> torch.nn.Module:
        return self._model
