import pytest
import torch
from unittest.mock import MagicMock
from core.base.io_types import TEXT_TO_TEXT, IOType, DataType


def _make_mock_model(loss_val: float = 1.5):
    model = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(4, 4, requires_grad=True)])
    model.forward.return_value = {"loss": torch.tensor(loss_val, requires_grad=True)}
    return model


def test_sft_registers():
    import adapters.llm.strategies.sft
    from core.registry import STRATEGIES
    assert "sft" in STRATEGIES.list_all()


def test_sft_compatible_io_types():
    from adapters.llm.strategies.sft import SFTStrategy
    assert TEXT_TO_TEXT in SFTStrategy.compatible_io_types


def test_sft_setup_returns_model_unchanged():
    from adapters.llm.strategies.sft import SFTStrategy
    model = _make_mock_model()
    result = SFTStrategy().setup(model, {})
    assert result is model


def test_sft_setup_enables_gradient_checkpointing():
    from adapters.llm.strategies.sft import SFTStrategy
    model = _make_mock_model()
    SFTStrategy().setup(model, {"gradient_checkpointing": True})
    model.raw_model.gradient_checkpointing_enable.assert_called_once()


def test_sft_setup_no_gradient_checkpointing_by_default():
    from adapters.llm.strategies.sft import SFTStrategy
    model = _make_mock_model()
    SFTStrategy().setup(model, {})
    model.raw_model.gradient_checkpointing_enable.assert_not_called()


def test_configure_optimizers_returns_adamw():
    from adapters.llm.strategies.sft import SFTStrategy
    from torch.optim import AdamW
    model = _make_mock_model()
    opt = SFTStrategy().configure_optimizers(model, {"lr": 2e-5})
    assert isinstance(opt, AdamW)


def test_training_step_returns_loss_tensor():
    from adapters.llm.strategies.sft import SFTStrategy
    model = _make_mock_model(loss_val=2.0)
    result = SFTStrategy().training_step(model, {}, step=0)
    assert "loss" in result
    assert isinstance(result["loss"], torch.Tensor)
    assert abs(result["loss"].item() - 2.0) < 1e-5


def test_teardown_is_noop():
    from adapters.llm.strategies.sft import SFTStrategy
    SFTStrategy().teardown(_make_mock_model())


def test_validate_rejects_image_io():
    from adapters.llm.strategies.sft import SFTStrategy
    with pytest.raises(ValueError, match="does not support io_type"):
        SFTStrategy().validate(IOType(DataType.IMAGE, DataType.CATEGORY), None)


def test_configure_scheduler_cosine():
    from adapters.llm.strategies.sft import SFTStrategy
    from torch.optim.lr_scheduler import CosineAnnealingLR
    params = [torch.randn(3, requires_grad=True)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = SFTStrategy().configure_scheduler(opt, {"scheduler": "cosine", "t_max": 3})
    assert isinstance(sched, CosineAnnealingLR)


def test_configure_scheduler_none_by_default():
    from adapters.llm.strategies.sft import SFTStrategy
    assert SFTStrategy().configure_scheduler(MagicMock(), {}) is None
