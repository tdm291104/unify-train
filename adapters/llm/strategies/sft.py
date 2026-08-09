from typing import Any
import torch
from torch.optim import AdamW
from core.base.model import BaseModel
from core.base.strategy import BaseStrategy
from core.base.io_types import TEXT_TO_TEXT
from core.registry import STRATEGIES


@STRATEGIES.register("sft")
class SFTStrategy(BaseStrategy):

    compatible_io_types = [TEXT_TO_TEXT]

    def setup(self, model: BaseModel, config: dict[str, Any]) -> BaseModel:
        if config.get("gradient_checkpointing", False):
            model.raw_model.gradient_checkpointing_enable()
        return model

    def configure_optimizers(self, model: BaseModel, config: dict[str, Any]) -> AdamW:
        lr = float(config.get("lr", 2e-5))
        weight_decay = float(config.get("weight_decay", 0.01))
        return AdamW(model.raw_model.parameters(), lr=lr, weight_decay=weight_decay)

    def configure_scheduler(self, optimizer: Any, config: dict[str, Any]) -> Any | None:
        name = config.get("scheduler")
        if name == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=int(config.get("t_max", 3)))
        if name == "step":
            from torch.optim.lr_scheduler import StepLR
            return StepLR(
                optimizer,
                step_size=int(config.get("step_size", 1)),
                gamma=float(config.get("gamma", 0.1)),
            )
        return None

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
