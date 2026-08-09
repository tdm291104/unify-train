from typing import Any
from core.base.evaluator import BaseEvaluator
from core.base.io_types import IMAGE_TO_CATEGORY
from core.registry import EVALUATORS


@EVALUATORS.register("accuracy")
class AccuracyEvaluator(BaseEvaluator):

    io_type = IMAGE_TO_CATEGORY

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> dict[str, float]:
        """Top-1 accuracy: predictions and references are lists of class indices."""
        if not references:
            return {"accuracy": 0.0}
        correct = sum(p == r for p, r in zip(predictions, references))
        return {"accuracy": correct / len(references)}
