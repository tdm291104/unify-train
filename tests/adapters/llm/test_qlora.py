import pytest
import torch
from unittest.mock import MagicMock, patch


def _make_mock_model():
    model = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(4, 4, requires_grad=True)])
    model.forward.return_value = {"loss": torch.tensor(1.0, requires_grad=True)}
    return model


def test_registers():
    import adapters.llm.strategies.qlora  # noqa: F401
    from core.registry import STRATEGIES
    assert "qlora" in STRATEGIES.list_all()


def test_setup_calls_prepare_before_lora():
    from adapters.llm.strategies.qlora import QLoRAStrategy

    mock_model = _make_mock_model()
    call_order = []

    def mock_prepare(raw_model):
        call_order.append("prepare")

    mock_peft_model = MagicMock()

    def mock_get_peft(raw_model, cfg):
        call_order.append("peft")
        return mock_peft_model

    with patch("adapters.llm.strategies.qlora.prepare_model_for_kbit_training", mock_prepare):
        with patch("adapters.llm.strategies.lora.get_peft_model", mock_get_peft):
            with patch("adapters.llm.strategies.lora.LoraConfig"):
                QLoRAStrategy().setup(mock_model, {"r": 8, "lora_alpha": 16})

    assert call_order == ["prepare", "peft"], f"Wrong call order: {call_order}"


def test_setup_missing_bitsandbytes_raises():
    from adapters.llm.strategies.qlora import QLoRAStrategy

    mock_model = _make_mock_model()

    with patch("adapters.llm.strategies.qlora.prepare_model_for_kbit_training",
               side_effect=ImportError("bitsandbytes")):
        with pytest.raises(ImportError, match="QLoRA requires bitsandbytes"):
            QLoRAStrategy().setup(mock_model, {})


def test_training_step_inherited():
    from adapters.llm.strategies.qlora import QLoRAStrategy
    mock_model = _make_mock_model()
    result = QLoRAStrategy().training_step(mock_model, {"x": torch.tensor([1])}, step=0)
    assert "loss" in result


def test_configure_optimizers_inherited():
    from adapters.llm.strategies.qlora import QLoRAStrategy
    from torch.optim import AdamW
    mock_model = _make_mock_model()
    opt = QLoRAStrategy().configure_optimizers(mock_model, {"lr": 1e-4})
    assert isinstance(opt, AdamW)
