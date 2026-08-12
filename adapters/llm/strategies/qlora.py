from typing import Any
from peft import prepare_model_for_kbit_training
from core.base.model import BaseModel
from core.registry import STRATEGIES
from adapters.llm.strategies.lora import LoRAStrategy


@STRATEGIES.register("qlora")
class QLoRAStrategy(LoRAStrategy):

    def setup(self, model: BaseModel, config: dict[str, Any]) -> BaseModel:
        try:
            prepare_model_for_kbit_training(model.raw_model)
        except Exception as exc:
            if "bitsandbytes" in str(exc).lower():
                raise ImportError(
                    "QLoRA requires bitsandbytes: pip install bitsandbytes"
                ) from exc
            raise
        return super().setup(model, config)
