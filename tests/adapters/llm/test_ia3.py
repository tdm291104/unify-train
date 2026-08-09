import pytest
import torch
from unittest.mock import MagicMock, patch
from core.base.io_types import TEXT_TO_TEXT, IOType, DataType


def _make_mock_model(loss_val: float = 1.5):
    model = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(4, 4, requires_grad=True)])
    model.forward.return_value = {"loss": torch.tensor(loss_val, requires_grad=True)}
    return model


def test_ia3_registers():
    import adapters.llm.strategies.ia3
    from core.registry import STRATEGIES
    assert "ia3" in STRATEGIES.list_all()


def test_ia3_compatible_io_types():
    from adapters.llm.strategies.ia3 import IA3Strategy
    assert TEXT_TO_TEXT in IA3Strategy.compatible_io_types


def test_ia3_setup_wraps_model_with_peft():
    from adapters.llm.strategies.ia3 import IA3Strategy
    model = _make_mock_model()
    peft_mock = MagicMock()
    with patch("adapters.llm.strategies.ia3.get_peft_model", return_value=peft_mock) as mock_peft, \
         patch("adapters.llm.strategies.ia3.IA3Config") as mock_cfg:
        result = IA3Strategy().setup(model, {})
    mock_peft.assert_called_once()
    assert result._model is peft_mock


def test_configure_optimizers_returns_adamw():
    from adapters.llm.strategies.ia3 import IA3Strategy
    from torch.optim import AdamW
    model = _make_mock_model()
    opt = IA3Strategy().configure_optimizers(model, {"lr": 1e-3})
    assert isinstance(opt, AdamW)


def test_training_step_returns_loss():
    from adapters.llm.strategies.ia3 import IA3Strategy
    model = _make_mock_model(loss_val=1.0)
    optimizer = MagicMock()
    result = IA3Strategy().training_step(model, {}, optimizer, step=0)
    assert "loss" in result
    assert abs(result["loss"] - 1.0) < 1e-5


def test_training_step_calls_optimizer():
    from adapters.llm.strategies.ia3 import IA3Strategy
    model = _make_mock_model()
    optimizer = MagicMock()
    IA3Strategy().training_step(model, {}, optimizer, step=0)
    optimizer.zero_grad.assert_called_once()
    optimizer.step.assert_called_once()


def test_teardown_merges_and_unloads():
    from adapters.llm.strategies.ia3 import IA3Strategy
    model = _make_mock_model()
    merged = MagicMock()
    model.raw_model.merge_and_unload.return_value = merged
    IA3Strategy().teardown(model)
    model.raw_model.merge_and_unload.assert_called_once()
    assert model._model is merged


def test_validate_rejects_image_io():
    from adapters.llm.strategies.ia3 import IA3Strategy
    with pytest.raises(ValueError, match="does not support io_type"):
        IA3Strategy().validate(IOType(DataType.IMAGE, DataType.CATEGORY), None)


def test_configure_scheduler_cosine():
    from adapters.llm.strategies.ia3 import IA3Strategy
    from torch.optim.lr_scheduler import CosineAnnealingLR
    params = [torch.randn(3, requires_grad=True)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = IA3Strategy().configure_scheduler(opt, {"scheduler": "cosine", "t_max": 5})
    assert isinstance(sched, CosineAnnealingLR)


def test_configure_scheduler_none_by_default():
    from adapters.llm.strategies.ia3 import IA3Strategy
    assert IA3Strategy().configure_scheduler(MagicMock(), {}) is None
