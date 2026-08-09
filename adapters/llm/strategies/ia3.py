from typing import Any
import torch
from peft import IA3Config, get_peft_model
from torch.optim import AdamW
from core.base.model import BaseModel
from core.base.strategy import BaseStrategy
from core.base.io_types import TEXT_TO_TEXT
from core.registry import STRATEGIES


@STRATEGIES.register("ia3")
class IA3Strategy(BaseStrategy):

    compatible_io_types = [TEXT_TO_TEXT]

    def setup(self, model: BaseModel, config: dict[str, Any]) -> BaseModel:
        ia3_cfg = IA3Config(
            task_type="CAUSAL_LM",
            feedforward_modules=config.get("feedforward_modules", []),
        )
        raw = model.raw_model
        peft_model = get_peft_model(raw, ia3_cfg)
        model._model = peft_model
        return model

    def configure_optimizers(self, model: BaseModel, config: dict[str, Any]) -> AdamW:
        lr = float(config.get("lr", 1e-3))
        return AdamW(filter(lambda p: p.requires_grad, model.raw_model.parameters()), lr=lr)

    def training_step(
        self,
        model: BaseModel,
        batch: dict[str, Any],
        step: int,
    ) -> dict[str, Any]:
        out = model.forward(batch)
        return {"loss": out["loss"]}

    def teardown(self, model: BaseModel) -> None:
        merged = model.raw_model.merge_and_unload()
        model._model = merged
