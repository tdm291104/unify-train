import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from core.registry import EVALUATORS
from adapters.llm.evaluators.classification import ClassificationEvaluator


class TestRegistration:
    def test_registers(self):
        assert "classification" in EVALUATORS.list_all()


class TestInit:
    def test_default_params(self):
        ev = ClassificationEvaluator(params=None)
        assert ev._metrics == {"f1", "precision", "recall", "exact_match"}
        assert ev._f1_average == "weighted"

    def test_custom_params(self):
        ev = ClassificationEvaluator(params={"metrics": ["f1"], "f1_average": "macro"})
        assert ev._metrics == {"f1"}
        assert ev._f1_average == "macro"


class TestExtract:
    def test_extract_shape(self):
        ev = ClassificationEvaluator()
        logits = torch.zeros(3, 5)
        labels = torch.zeros(3, dtype=torch.long)
        preds, refs = ev.extract({"logits": logits}, {"labels": labels})
        assert len(preds) == 3
        assert len(refs) == 3

    def test_extract_argmax(self):
        ev = ClassificationEvaluator()
        # 3 samples, 5 classes — class 2 has highest logit for all
        logits = torch.zeros(3, 5)
        logits[:, 2] = 10.0
        labels = torch.zeros(3, dtype=torch.long)
        preds, refs = ev.extract({"logits": logits}, {"labels": labels})
        assert all(p == 2 for p in preds)


class TestCompute:
    def _mock_sklearn(self, prec=1.0, rec=1.0, f1=1.0):
        """Return a patch.dict context manager with mocked sklearn."""
        mock_sklearn = MagicMock()
        mock_metrics = MagicMock()
        mock_sklearn.metrics = mock_metrics
        mock_metrics.precision_recall_fscore_support.return_value = (prec, rec, f1, None)
        return mock_sklearn, mock_metrics

    def test_compute_all_correct(self):
        ev = ClassificationEvaluator()
        mock_sklearn, mock_metrics = self._mock_sklearn(prec=1.0, rec=1.0, f1=1.0)
        preds = [0, 1, 2]
        refs = [0, 1, 2]
        with patch.dict(sys.modules, {"sklearn": mock_sklearn, "sklearn.metrics": mock_metrics}):
            result = ev.compute(preds, refs)
        assert pytest.approx(result["f1"]) == 1.0
        assert pytest.approx(result["precision"]) == 1.0
        assert pytest.approx(result["recall"]) == 1.0
        assert pytest.approx(result["exact_match"]) == 1.0

    def test_compute_only_configured_keys_returned(self):
        ev = ClassificationEvaluator(params={"metrics": ["f1"]})
        mock_sklearn, mock_metrics = self._mock_sklearn(prec=0.8, rec=0.7, f1=0.75)
        preds = [0, 1, 2]
        refs = [0, 1, 1]
        with patch.dict(sys.modules, {"sklearn": mock_sklearn, "sklearn.metrics": mock_metrics}):
            result = ev.compute(preds, refs)
        assert set(result.keys()) == {"f1"}

    def test_exact_match_partial(self):
        ev = ClassificationEvaluator(params={"metrics": ["exact_match"]})
        mock_sklearn, mock_metrics = self._mock_sklearn()
        preds = [0, 1, 0, 1]
        refs = [0, 1, 1, 0]  # 2 out of 4 correct
        with patch.dict(sys.modules, {"sklearn": mock_sklearn, "sklearn.metrics": mock_metrics}):
            result = ev.compute(preds, refs)
        assert pytest.approx(result["exact_match"]) == 0.5
        assert "f1" not in result
        assert "precision" not in result
        assert "recall" not in result

    def test_missing_sklearn(self):
        ev = ClassificationEvaluator()
        with patch.dict(sys.modules, {"sklearn": None, "sklearn.metrics": None}):
            with pytest.raises(ImportError, match="scikit-learn"):
                ev.compute([0, 1], [0, 1])
