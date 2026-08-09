import pytest
import torch
from unittest.mock import MagicMock, patch
from core.base.io_types import TEXT_TO_TEXT


def test_distilgpt2_registers_under_correct_name():
    import adapters.llm.models.distilgpt2
    from core.registry import MODELS
    assert "distilgpt2" in MODELS.list_all()


def test_distilgpt2_io_type_is_text_to_text():
    import adapters.llm.models.distilgpt2
    from core.registry import MODELS
    assert MODELS.get("distilgpt2").io_type == TEXT_TO_TEXT


def test_distilgpt2_build_calls_from_pretrained():
    from adapters.llm.models.distilgpt2 import DistilGPT2Model

    mock_tok = MagicMock()
    mock_hf = MagicMock()

    with patch("adapters.llm.models.distilgpt2.AutoTokenizer") as MockTok:
        with patch("adapters.llm.models.distilgpt2.AutoModelForCausalLM") as MockHF:
            MockTok.from_pretrained.return_value = mock_tok
            MockHF.from_pretrained.return_value = mock_hf

            m = DistilGPT2Model()
            m.build({"pretrained": "distilgpt2"})

            MockTok.from_pretrained.assert_called_once_with("distilgpt2")
            MockHF.from_pretrained.assert_called_once_with("distilgpt2")
            assert mock_tok.pad_token == mock_tok.eos_token


def test_distilgpt2_forward_returns_loss_and_logits():
    from adapters.llm.models.distilgpt2 import DistilGPT2Model

    mock_tok = MagicMock()
    mock_outputs = MagicMock(loss=torch.tensor(1.5), logits=torch.zeros(1, 3, 50257))
    mock_hf = MagicMock(return_value=mock_outputs)

    with patch("adapters.llm.models.distilgpt2.AutoTokenizer") as MockTok:
        with patch("adapters.llm.models.distilgpt2.AutoModelForCausalLM") as MockHF:
            MockTok.from_pretrained.return_value = mock_tok
            MockHF.from_pretrained.return_value = mock_hf

            m = DistilGPT2Model()
            m.build({"pretrained": "distilgpt2"})

            batch = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "labels": torch.tensor([[1, 2, 3]]),
            }
            out = m.forward(batch)

    assert "loss" in out
    assert "logits" in out
    assert out["loss"] is mock_outputs.loss


def test_distilgpt2_raw_model_property():
    from adapters.llm.models.distilgpt2 import DistilGPT2Model

    mock_hf = MagicMock()
    with patch("adapters.llm.models.distilgpt2.AutoTokenizer"):
        with patch("adapters.llm.models.distilgpt2.AutoModelForCausalLM") as MockHF:
            MockHF.from_pretrained.return_value = mock_hf
            m = DistilGPT2Model()
            m.build({"pretrained": "distilgpt2"})
            assert m.raw_model is mock_hf


def test_distilgpt2_save_calls_save_pretrained(tmp_path):
    from adapters.llm.models.distilgpt2 import DistilGPT2Model

    mock_tok = MagicMock()
    mock_hf = MagicMock()

    with patch("adapters.llm.models.distilgpt2.AutoTokenizer") as MockTok:
        with patch("adapters.llm.models.distilgpt2.AutoModelForCausalLM") as MockHF:
            MockTok.from_pretrained.return_value = mock_tok
            MockHF.from_pretrained.return_value = mock_hf

            m = DistilGPT2Model()
            m.build({"pretrained": "distilgpt2"})
            m.save(str(tmp_path))

    mock_hf.save_pretrained.assert_called_once_with(str(tmp_path))
    mock_tok.save_pretrained.assert_called_once_with(str(tmp_path))
