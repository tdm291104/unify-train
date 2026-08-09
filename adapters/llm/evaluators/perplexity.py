import math
from typing import Any
from core.base.evaluator import BaseEvaluator
from core.base.io_types import TEXT_TO_TEXT
from core.registry import EVALUATORS


@EVALUATORS.register("perplexity")
class PerplexityEvaluator(BaseEvaluator):

    io_type = TEXT_TO_TEXT

    def extract(
        self,
        model_output: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple[list[Any], list[Any]]:
        return [model_output["loss"].item()], []

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> dict[str, float]:
        """Compute perplexity from per-token cross-entropy losses."""
        if not predictions:
            return {"perplexity": float("inf")}
        avg_loss = sum(predictions) / len(predictions)
        return {"perplexity": math.exp(avg_loss)}
