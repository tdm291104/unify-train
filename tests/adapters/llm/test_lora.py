import pytest
import torch
from unittest.mock import MagicMock, patch, call
from core.base.io_types import TEXT_TO_TEXT, IOType, DataType


def _make_mock_model(loss_val: float = 1.5):
    model = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(4, 4, requires_grad=True)])
    model.forward.return_value = {"loss": torch.tensor(loss_val, requires_grad=True)}
    return model


def test_lora_registers():
    import adapters.llm.strategies.lora
    from core.registry import STRATEGIES
    assert "lora" in STRATEGIES.list_all()


def test_lora_compatible_io_types():
    import adapters.llm.strategies.lora
    from core.registry import STRATEGIES
    strat = STRATEGIES.get("lora")
    assert TEXT_TO_TEXT in strat.compatible_io_types


def test_lora_setup_calls_get_peft_model():
    from adapters.llm.strategies.lora import LoRAStrategy

    mock_peft = MagicMock()
    mock_model = _make_mock_model()

    with patch("adapters.llm.strategies.lora.get_peft_model", return_value=mock_peft) as mock_gpm:
        with patch("adapters.llm.strategies.lora.LoraConfig") as MockCfg:
            strat = LoRAStrategy()
            result = strat.setup(mock_model, {"r": 16, "lora_alpha": 32})

    MockCfg.assert_called_once_with(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    mock_gpm.assert_called_once_with(mock_model.raw_model, MockCfg.return_value)
    assert mock_model._model is mock_peft
    assert result is mock_model


def test_lora_setup_defaults():
    from adapters.llm.strategies.lora import LoRAStrategy

    mock_model = _make_mock_model()
    with patch("adapters.llm.strategies.lora.get_peft_model") as mock_gpm:
        with patch("adapters.llm.strategies.lora.LoraConfig") as MockCfg:
            LoRAStrategy().setup(mock_model, {})
    MockCfg.assert_called_once_with(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")


def test_configure_optimizers_returns_adamw():
    from adapters.llm.strategies.lora import LoRAStrategy
    from torch.optim import AdamW

    mock_model = _make_mock_model()
    strat = LoRAStrategy()
    opt = strat.configure_optimizers(mock_model, {"lr": 1e-4})
    assert isinstance(opt, AdamW)


def test_training_step_returns_loss_tensor():
    from adapters.llm.strategies.lora import LoRAStrategy

    mock_model = _make_mock_model(loss_val=2.5)
    result = LoRAStrategy().training_step(mock_model, {"input_ids": torch.tensor([[1]])}, step=0)
    assert "loss" in result
    assert isinstance(result["loss"], torch.Tensor)
    assert abs(result["loss"].item() - 2.5) < 1e-5


def test_teardown_calls_merge_and_unload():
    from adapters.llm.strategies.lora import LoRAStrategy

    merged = MagicMock()
    mock_model = _make_mock_model()
    mock_model.raw_model.merge_and_unload.return_value = merged

    LoRAStrategy().teardown(mock_model)

    mock_model.raw_model.merge_and_unload.assert_called_once()
    assert mock_model._model is merged


def test_validate_rejects_wrong_io_type():
    from adapters.llm.strategies.lora import LoRAStrategy

    strat = LoRAStrategy()
    image_to_cat = IOType(DataType.IMAGE, DataType.CATEGORY)
    with pytest.raises(ValueError, match="does not support io_type"):
        strat.validate(image_to_cat, None)


def test_validate_accepts_text_to_text():
    from adapters.llm.strategies.lora import LoRAStrategy

    LoRAStrategy().validate(TEXT_TO_TEXT, None)
