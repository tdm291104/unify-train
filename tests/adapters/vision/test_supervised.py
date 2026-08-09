import pytest
import torch
from unittest.mock import MagicMock
from core.base.io_types import IMAGE_TO_CATEGORY, IOType, DataType


def _make_mock_model(loss_val: float = 1.2):
    model = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(4, 4, requires_grad=True)])
    model.forward.return_value = {"loss": torch.tensor(loss_val, requires_grad=True), "logits": torch.randn(2, 3)}
    return model


def test_supervised_registers():
    import adapters.vision.strategies.supervised
    from core.registry import STRATEGIES
    assert "supervised" in STRATEGIES.list_all()


def test_supervised_compatible_io_types():
    import adapters.vision.strategies.supervised
    from core.registry import STRATEGIES
    strat = STRATEGIES.get("supervised")
    assert IMAGE_TO_CATEGORY in strat.compatible_io_types


def test_supervised_setup_returns_model_unchanged():
    from adapters.vision.strategies.supervised import SupervisedStrategy
    model = _make_mock_model()
    result = SupervisedStrategy().setup(model, {})
    assert result is model


def test_configure_optimizers_returns_sgd():
    from adapters.vision.strategies.supervised import SupervisedStrategy
    from torch.optim import SGD
    model = _make_mock_model()
    opt = SupervisedStrategy().configure_optimizers(model, {"lr": 0.01})
    assert isinstance(opt, SGD)


def test_training_step_returns_loss_tensor():
    from adapters.vision.strategies.supervised import SupervisedStrategy
    model = _make_mock_model(loss_val=0.8)
    result = SupervisedStrategy().training_step(model, {}, step=0)
    assert "loss" in result
    assert isinstance(result["loss"], torch.Tensor)
    assert abs(result["loss"].item() - 0.8) < 1e-5


def test_teardown_is_noop():
    from adapters.vision.strategies.supervised import SupervisedStrategy
    model = _make_mock_model()
    SupervisedStrategy().teardown(model)  # should not raise


def test_validate_rejects_text_to_text():
    from adapters.vision.strategies.supervised import SupervisedStrategy
    from core.base.io_types import TEXT_TO_TEXT
    with pytest.raises(ValueError, match="does not support io_type"):
        SupervisedStrategy().validate(TEXT_TO_TEXT, None)


def test_validate_accepts_image_to_category():
    from adapters.vision.strategies.supervised import SupervisedStrategy
    SupervisedStrategy().validate(IMAGE_TO_CATEGORY, None)
