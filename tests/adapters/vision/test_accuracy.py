import pytest
from core.base.io_types import IMAGE_TO_CATEGORY


def test_accuracy_registers():
    import adapters.vision.evaluators.accuracy
    from core.registry import EVALUATORS
    assert "accuracy" in EVALUATORS.list_all()


def test_accuracy_io_type():
    import adapters.vision.evaluators.accuracy
    from core.registry import EVALUATORS
    assert EVALUATORS.get("accuracy").io_type == IMAGE_TO_CATEGORY


def test_accuracy_empty_returns_zero():
    from adapters.vision.evaluators.accuracy import AccuracyEvaluator
    result = AccuracyEvaluator().compute([], [])
    assert result["accuracy"] == 0.0


def test_accuracy_all_correct():
    from adapters.vision.evaluators.accuracy import AccuracyEvaluator
    result = AccuracyEvaluator().compute([0, 1, 2], [0, 1, 2])
    assert result["accuracy"] == 1.0


def test_accuracy_none_correct():
    from adapters.vision.evaluators.accuracy import AccuracyEvaluator
    result = AccuracyEvaluator().compute([0, 0, 0], [1, 2, 3])
    assert result["accuracy"] == 0.0


def test_accuracy_partial():
    from adapters.vision.evaluators.accuracy import AccuracyEvaluator
    result = AccuracyEvaluator().compute([0, 1, 0, 3], [0, 1, 2, 3])
    assert abs(result["accuracy"] - 0.75) < 1e-6


def test_vision_init_registers_all():
    import adapters.vision
    from core.registry import MODELS, DATASETS, STRATEGIES, EVALUATORS
    assert "resnet18" in MODELS.list_all()
    assert "synthetic_image" in DATASETS.list_all()
    assert "supervised" in STRATEGIES.list_all()
    assert "accuracy" in EVALUATORS.list_all()
