from typing import Any
from core.base.evaluator import BaseEvaluator
from core.base.io_types import TEXT_TO_TEXT
from core.registry import EVALUATORS


@EVALUATORS.register("generation")
class GenerationEvaluator(BaseEvaluator):

    io_type = TEXT_TO_TEXT

    def __init__(self, params=None) -> None:
        params = params or {}
        self._metrics = set(params.get("metrics", ["bleu", "rouge_l"]))

    def extract(
        self,
        model_output: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple[list[Any], list[Any]]:
        logits = model_output["logits"]       # [B, T, V]
        pred_ids = logits.argmax(dim=-1)      # [B, T]
        labels = batch["labels"]              # [B, T]

        # Keep only response positions (where labels != -100) for meaningful metrics
        preds, refs = [], []
        for pred_row, label_row in zip(pred_ids, labels):
            mask = label_row != -100
            preds.append(pred_row[mask].tolist())
            refs.append(label_row[mask].tolist())

        return preds, refs

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> dict[str, float]:
        # predictions, references: list[list[int]]
        # Convert to space-separated strings for metric computation
        pred_strs = [" ".join(str(t) for t in p) for p in predictions]
        ref_strs = [" ".join(str(t) for t in r) for r in references]

        result: dict[str, float] = {}

        if "bleu" in self._metrics:
            try:
                import sacrebleu
            except ImportError as exc:
                raise ImportError(
                    "GenerationEvaluator with BLEU requires sacrebleu: pip install sacrebleu"
                ) from exc
            score = sacrebleu.corpus_bleu(pred_strs, [ref_strs])
            result["bleu"] = score.score

        if "rouge_l" in self._metrics:
            try:
                from rouge_score import rouge_scorer as rs
            except ImportError as exc:
                raise ImportError(
                    "GenerationEvaluator with ROUGE-L requires rouge-score: pip install rouge-score"
                ) from exc
            scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
            scores = [
                scorer.score(r, p)["rougeL"].fmeasure
                for p, r in zip(pred_strs, ref_strs)
            ]
            result["rouge_l"] = sum(scores) / len(scores) if scores else 0.0

        return result
