from typing import Any
import torch
from torch.optim import SGD
from core.base.model import BaseModel
from core.base.strategy import BaseStrategy
from core.base.io_types import IMAGE_TO_CATEGORY
from core.registry import STRATEGIES


@STRATEGIES.register("supervised")
class SupervisedStrategy(BaseStrategy):

    compatible_io_types = [IMAGE_TO_CATEGORY]

    def setup(self, model: BaseModel, config: dict[str, Any]) -> BaseModel:
        return model

    def configure_optimizers(self, model: BaseModel, config: dict[str, Any]) -> SGD:
        lr = float(config.get("lr", 1e-2))
        momentum = float(config.get("momentum", 0.9))
        return SGD(model.raw_model.parameters(), lr=lr, momentum=momentum)

    def training_step(
        self,
        model: BaseModel,
        batch: dict[str, Any],
        step: int,
    ) -> dict[str, Any]:
        out = model.forward(batch)
        return {"loss": out["loss"]}

    def teardown(self, model: BaseModel) -> None:
        pass
