from typing import Any
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from core.base.model import BaseModel
from core.base.strategy import BaseStrategy
from core.base.io_types import TEXT_TO_TEXT
from core.registry import STRATEGIES


@STRATEGIES.register("orpo")
class ORPOStrategy(BaseStrategy):

    compatible_io_types = [TEXT_TO_TEXT]

    def __init__(self) -> None:
        self._lambda_orpo: float = 0.1

    def setup(self, model: BaseModel, config: dict[str, Any]) -> BaseModel:
        self._lambda_orpo = float(config.get("lambda_orpo", 0.1))
        if config.get("gradient_checkpointing", False):
            model.raw_model.gradient_checkpointing_enable()
        return model

    def configure_optimizers(self, model: BaseModel, config: dict[str, Any]) -> AdamW:
        lr = float(config.get("lr", 8e-6))
        return AdamW(model.raw_model.parameters(), lr=lr)

    def _log_odds(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-sample log odds for non-padding tokens."""
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        shifted_labels = labels[:, 1:].clone()
        mask = shifted_labels != -100
        shifted_labels[~mask] = 0
        gathered = log_probs.gather(2, shifted_labels.unsqueeze(-1)).squeeze(-1)
        log_p = (gathered * mask.float()).sum(-1)
        log_1_minus_p = torch.log1p(-torch.exp(log_p.clamp(max=-1e-7)))
        return log_p - log_1_minus_p

    def training_step(
        self,
        model: BaseModel,
        batch: dict[str, Any],
        step: int,
    ) -> dict[str, Any]:
        chosen = batch["chosen"]
        rejected = batch["rejected"]

        chosen_out = model.forward(chosen)
        sft_loss: torch.Tensor = chosen_out["loss"]
        chosen_log_odds = self._log_odds(chosen_out["logits"], chosen["labels"])

        rejected_out = model.forward(rejected)
        rejected_log_odds = self._log_odds(rejected_out["logits"], rejected["labels"])

        or_loss = -F.logsigmoid(chosen_log_odds - rejected_log_odds).mean()
        loss = sft_loss + self._lambda_orpo * or_loss

        return {
            "loss": loss,
            "sft_loss": sft_loss.detach().item(),
            "or_loss": or_loss.detach().item(),
        }

    def teardown(self, model: BaseModel) -> None:
        pass
