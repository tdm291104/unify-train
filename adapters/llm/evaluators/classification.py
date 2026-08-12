from typing import Any

from core.base.evaluator import BaseEvaluator
from core.base.io_types import TEXT_TO_TEXT
from core.registry import EVALUATORS


@EVALUATORS.register("classification")
class ClassificationEvaluator(BaseEvaluator):

    io_type = TEXT_TO_TEXT

    def __init__(self, params=None) -> None:
        params = params or {}
        self._metrics = set(params.get("metrics", ["f1", "precision", "recall", "exact_match"]))
        self._f1_average = params.get("f1_average", "weighted")

    def extract(
        self,
        model_output: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple[list[int], list[int]]:
        logits = model_output["logits"]
        labels = batch["labels"]

        # For causal LM logits [B, T, V], use last-position logits as class scores
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # [B, V]
        preds = logits.argmax(dim=-1).tolist()

        # For sequence labels [B, T], take the last non-masked label per sample
        if labels.dim() == 2:
            refs = []
            for row in labels:
                valid = row[row != -100]
                refs.append(valid[-1].item() if valid.numel() > 0 else 0)
        else:
            refs = labels.tolist()

        return preds, refs

    def compute(
        self,
        predictions: list[int],
        references: list[int],
    ) -> dict[str, float]:
        try:
            from sklearn.metrics import precision_recall_fscore_support
        except ImportError as exc:
            raise ImportError(
                "ClassificationEvaluator requires scikit-learn: pip install scikit-learn"
            ) from exc

        result: dict[str, float] = {}

        if any(m in self._metrics for m in ("f1", "precision", "recall")):
            prec, rec, f1, _ = precision_recall_fscore_support(
                references, predictions, average=self._f1_average, zero_division=0
            )
            if "precision" in self._metrics:
                result["precision"] = float(prec)
            if "recall" in self._metrics:
                result["recall"] = float(rec)
            if "f1" in self._metrics:
                result["f1"] = float(f1)

        if "exact_match" in self._metrics:
            result["exact_match"] = (
                sum(p == r for p, r in zip(predictions, references)) / len(predictions)
                if predictions
                else 0.0
            )

        return result
