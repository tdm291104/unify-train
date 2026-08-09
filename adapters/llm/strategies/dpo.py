from typing import Any
import copy
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from core.base.model import BaseModel
from core.base.strategy import BaseStrategy
from core.base.io_types import TEXT_TO_TEXT
from core.registry import STRATEGIES


@STRATEGIES.register("dpo")
class DPOStrategy(BaseStrategy):

    compatible_io_types = [TEXT_TO_TEXT]

    def setup(self, model: BaseModel, config: dict[str, Any]) -> BaseModel:
        # Freeze a deep copy as the reference model before applying PEFT
        self._ref_model = copy.deepcopy(model.raw_model)
        for p in self._ref_model.parameters():
            p.requires_grad_(False)
        self._ref_model.eval()

        self._beta = float(config.get("beta", 0.1))

        lora_cfg = LoraConfig(
            r=config.get("r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias=config.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model._model = get_peft_model(model.raw_model, lora_cfg)
        return model

    def configure_optimizers(self, model: BaseModel, config: dict[str, Any]) -> AdamW:
        lr = float(config.get("lr", 1e-4))
        return AdamW(filter(lambda p: p.requires_grad, model.raw_model.parameters()), lr=lr)

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

    def _log_probs(self, model_out_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-sample sum of log-probs for non-padding tokens."""
        # logits: (B, T, V), labels: (B, T)  (-100 = ignore)
        log_probs = F.log_softmax(model_out_logits[:, :-1, :], dim=-1)
        shifted_labels = labels[:, 1:].clone()
        mask = shifted_labels != -100
        shifted_labels[~mask] = 0
        gathered = log_probs.gather(2, shifted_labels.unsqueeze(-1)).squeeze(-1)
        return (gathered * mask.float()).sum(-1)

    def training_step(
        self,
        model: BaseModel,
        batch: dict[str, Any],
        step: int,
    ) -> dict[str, Any]:
        chosen = batch["chosen"]
        rejected = batch["rejected"]

        # Policy log-probs
        chosen_out = model.forward(chosen)
        policy_chosen_lp = self._log_probs(chosen_out["logits"], chosen["labels"])

        rejected_out = model.forward(rejected)
        policy_rejected_lp = self._log_probs(rejected_out["logits"], rejected["labels"])

        # Reference log-probs (no grad)
        with torch.no_grad():
            ref_chosen_lp = self._log_probs(
                self._ref_model(input_ids=chosen["input_ids"], attention_mask=chosen["attention_mask"]).logits,
                chosen["labels"],
            )
            ref_rejected_lp = self._log_probs(
                self._ref_model(input_ids=rejected["input_ids"], attention_mask=rejected["attention_mask"]).logits,
                rejected["labels"],
            )

        # DPO loss
        ratio_chosen = policy_chosen_lp - ref_chosen_lp
        ratio_rejected = policy_rejected_lp - ref_rejected_lp
        loss = -F.logsigmoid(self._beta * (ratio_chosen - ratio_rejected)).mean()
        return {"loss": loss}

    def teardown(self, model: BaseModel) -> None:
        merged = model.raw_model.merge_and_unload()
        model._model = merged
        if hasattr(self, "_ref_model"):
            del self._ref_model
