import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from core.registry import EVALUATORS
from adapters.llm.evaluators.generation import GenerationEvaluator


class TestRegistration:
    def test_registers(self):
        assert "generation" in EVALUATORS.list_all()


class TestInit:
    def test_default_metrics(self):
        ev = GenerationEvaluator(params=None)
        assert ev._metrics == {"bleu", "rouge_l"}

    def test_custom_metrics(self):
        ev = GenerationEvaluator(params={"metrics": ["bleu"]})
        assert ev._metrics == {"bleu"}


class TestExtract:
    def _make_inputs(self, B=2, T=5, V=10):
        logits = torch.zeros(B, T, V)
        labels = torch.zeros(B, T, dtype=torch.long)
        return logits, labels

    def test_extract_shape(self):
        ev = GenerationEvaluator()
        logits, labels = self._make_inputs(B=2, T=5, V=10)
        out = {"logits": logits}
        batch = {"labels": labels}
        preds, refs = ev.extract(out, batch)
        assert len(preds) == 2
        assert len(refs) == 2

    def test_extract_argmax(self):
        ev = GenerationEvaluator()
        # B=1, T=1, V=10 — set token index 3 as highest
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 3] = 10.0
        labels = torch.zeros(1, 1, dtype=torch.long)
        preds, refs = ev.extract({"logits": logits}, {"labels": labels})
        assert preds[0][0] == 3

    def test_extract_labels_minus100_replaced(self):
        ev = GenerationEvaluator()
        logits = torch.zeros(2, 3, 5)
        labels = torch.tensor([[-100, 1, 2], [3, -100, 5]])
        preds, refs = ev.extract({"logits": logits}, {"labels": labels})
        # -100 positions should become 0
        assert refs[0][0] == 0
        assert refs[1][1] == 0
        # Non-masked positions should remain
        assert refs[0][1] == 1
        assert refs[1][0] == 3


class TestCompute:
    def _ev_bleu_only(self):
        ev = GenerationEvaluator(params={"metrics": ["bleu"]})
        return ev

    def _ev_rouge_only(self):
        ev = GenerationEvaluator(params={"metrics": ["rouge_l"]})
        return ev

    def test_compute_bleu_only(self):
        ev = self._ev_bleu_only()
        preds = [[1, 2, 3], [4, 5, 6]]
        refs = [[1, 2, 3], [4, 5, 6]]

        mock_sacrebleu = MagicMock()
        mock_score = MagicMock()
        mock_score.score = 99.0
        mock_sacrebleu.corpus_bleu.return_value = mock_score

        with patch.dict(sys.modules, {"sacrebleu": mock_sacrebleu}):
            result = ev.compute(preds, refs)

        pred_strs = ["1 2 3", "4 5 6"]
        ref_strs = ["1 2 3", "4 5 6"]
        mock_sacrebleu.corpus_bleu.assert_called_once_with(pred_strs, [ref_strs])
        assert result == {"bleu": 99.0}

    def test_compute_rouge_only(self):
        ev = self._ev_rouge_only()
        preds = [[1, 2, 3]]
        refs = [[1, 2, 3]]

        mock_rouge_score = MagicMock()
        mock_rs = MagicMock()
        mock_scorer_instance = MagicMock()
        mock_rouge_score.rouge_scorer = mock_rs
        mock_rs.RougeScorer.return_value = mock_scorer_instance

        # scorer.score(r, p)["rougeL"].fmeasure
        mock_score_val = MagicMock()
        mock_score_val.fmeasure = 0.75
        mock_scorer_instance.score.return_value = {"rougeL": mock_score_val}

        with patch.dict(
            sys.modules,
            {
                "rouge_score": mock_rouge_score,
                "rouge_score.rouge_scorer": mock_rs,
            },
        ):
            result = ev.compute(preds, refs)

        mock_rs.RougeScorer.assert_called_once_with(["rougeL"], use_stemmer=False)
        assert result == {"rouge_l": 0.75}

    def test_compute_missing_sacrebleu(self):
        ev = self._ev_bleu_only()
        with patch.dict(sys.modules, {"sacrebleu": None}):
            try:
                import importlib
                import adapters.llm.evaluators.generation as gen_mod
                importlib.reload(gen_mod)
            except Exception:
                pass
            with pytest_raises_import_error("sacrebleu"):
                ev.compute([[1, 2]], [[1, 2]])

    def test_compute_missing_rouge_score(self):
        ev = self._ev_rouge_only()
        with patch.dict(sys.modules, {"rouge_score": None, "rouge_score.rouge_scorer": None}):
            with pytest_raises_import_error("rouge-score"):
                ev.compute([[1, 2]], [[1, 2]])


import contextlib
import pytest


@contextlib.contextmanager
def pytest_raises_import_error(fragment: str):
    with pytest.raises(ImportError, match=fragment):
        yield
