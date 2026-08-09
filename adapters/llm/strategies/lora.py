from typing import Any
import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from core.base.model import BaseModel
from core.base.strategy import BaseStrategy
from core.base.io_types import TEXT_TO_TEXT
from core.registry import STRATEGIES


@STRATEGIES.register("lora")
class LoRAStrategy(BaseStrategy):

    compatible_io_types = [TEXT_TO_TEXT]

    def setup(self, model: BaseModel, config: dict[str, Any]) -> BaseModel:
        lora_cfg = LoraConfig(
            r=config.get("r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias=config.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        raw = model.raw_model
        peft_model = get_peft_model(raw, lora_cfg)
        model._model = peft_model
        return model

    def configure_optimizers(self, model: BaseModel, config: dict[str, Any]) -> AdamW:
        lr = float(config.get("lr", 3e-4))
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
