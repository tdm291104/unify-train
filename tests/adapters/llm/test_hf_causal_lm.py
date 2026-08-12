import torch
from unittest.mock import MagicMock, patch


def _mock_tokenizer():
    tok = MagicMock()
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0
    return tok


def _mock_auto_model():
    m = MagicMock()
    out = MagicMock()
    out.loss = torch.tensor(1.0, requires_grad=True)
    out.logits = torch.zeros(2, 4, 100)
    m.return_value = out
    return m


def test_registers():
    import adapters.llm.models.hf_causal_lm  # noqa: F401
    from core.registry import MODELS
    assert "hf_causal_lm" in MODELS.list_all()


def test_build_calls_auto_classes():
    from adapters.llm.models.hf_causal_lm import HFCausalLMModel
    with patch("adapters.llm.models.hf_causal_lm.AutoModelForCausalLM") as MockModel:
        with patch("adapters.llm.models.hf_causal_lm.AutoTokenizer") as MockTok:
            MockModel.from_pretrained.return_value = MagicMock()
            MockTok.from_pretrained.return_value = _mock_tokenizer()
            m = HFCausalLMModel()
            m.build({"pretrained": "gpt2"})
    MockModel.from_pretrained.assert_called_once_with("gpt2", quantization_config=None)
    MockTok.from_pretrained.assert_called_once_with("gpt2")


def test_build_with_4bit_creates_bnb_config():
    from adapters.llm.models.hf_causal_lm import HFCausalLMModel
    mock_bnb_instance = MagicMock()
    with patch("adapters.llm.models.hf_causal_lm.AutoModelForCausalLM") as MockModel:
        with patch("adapters.llm.models.hf_causal_lm.AutoTokenizer") as MockTok:
            with patch("adapters.llm.models.hf_causal_lm.BitsAndBytesConfig", return_value=mock_bnb_instance) as MockBnB:
                MockModel.from_pretrained.return_value = MagicMock()
                MockTok.from_pretrained.return_value = _mock_tokenizer()
                m = HFCausalLMModel()
                m.build({"pretrained": "gpt2", "load_in_4bit": True})
    MockBnB.assert_called_once()
    MockModel.from_pretrained.assert_called_once_with("gpt2", quantization_config=mock_bnb_instance)


def test_forward_returns_loss_and_logits():
    from adapters.llm.models.hf_causal_lm import HFCausalLMModel
    mock_raw = _mock_auto_model()
    with patch("adapters.llm.models.hf_causal_lm.AutoModelForCausalLM") as MockModel:
        with patch("adapters.llm.models.hf_causal_lm.AutoTokenizer") as MockTok:
            MockModel.from_pretrained.return_value = mock_raw
            MockTok.from_pretrained.return_value = _mock_tokenizer()
            m = HFCausalLMModel()
            m.build({"pretrained": "gpt2"})
    batch = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.ones(2, 4, dtype=torch.long),
    }
    out = m.forward(batch)
    assert "loss" in out
    assert "logits" in out
    assert out["loss"] is mock_raw.return_value.loss
    assert out["logits"] is mock_raw.return_value.logits


def test_raw_model_property():
    from adapters.llm.models.hf_causal_lm import HFCausalLMModel
    mock_raw = MagicMock()
    with patch("adapters.llm.models.hf_causal_lm.AutoModelForCausalLM") as MockModel:
        with patch("adapters.llm.models.hf_causal_lm.AutoTokenizer") as MockTok:
            MockModel.from_pretrained.return_value = mock_raw
            MockTok.from_pretrained.return_value = _mock_tokenizer()
            m = HFCausalLMModel()
            m.build({"pretrained": "gpt2"})
    assert m.raw_model is mock_raw


def test_save_calls_save_pretrained(tmp_path):
    from adapters.llm.models.hf_causal_lm import HFCausalLMModel
    mock_raw = MagicMock()
    tok = _mock_tokenizer()
    with patch("adapters.llm.models.hf_causal_lm.AutoModelForCausalLM") as MockModel:
        with patch("adapters.llm.models.hf_causal_lm.AutoTokenizer") as MockTok:
            MockModel.from_pretrained.return_value = mock_raw
            MockTok.from_pretrained.return_value = tok
            m = HFCausalLMModel()
            m.build({"pretrained": "gpt2"})
    m.save(str(tmp_path))
    mock_raw.save_pretrained.assert_called_once_with(str(tmp_path))
    tok.save_pretrained.assert_called_once_with(str(tmp_path))


def test_load_calls_from_pretrained():
    from adapters.llm.models.hf_causal_lm import HFCausalLMModel
    mock_raw = MagicMock()
    tok = _mock_tokenizer()
    with patch("adapters.llm.models.hf_causal_lm.AutoModelForCausalLM") as MockModel:
        with patch("adapters.llm.models.hf_causal_lm.AutoTokenizer") as MockTok:
            MockModel.from_pretrained.return_value = mock_raw
            MockTok.from_pretrained.return_value = tok
            m = HFCausalLMModel()
            m.load("/some/path")
    MockModel.from_pretrained.assert_called_once_with("/some/path")
    MockTok.from_pretrained.assert_called_once_with("/some/path")
    assert tok.pad_token == tok.eos_token


def test_tokenizer_property():
    from adapters.llm.models.hf_causal_lm import HFCausalLMModel
    tok = _mock_tokenizer()
    with patch("adapters.llm.models.hf_causal_lm.AutoModelForCausalLM") as MockModel:
        with patch("adapters.llm.models.hf_causal_lm.AutoTokenizer") as MockTok:
            MockModel.from_pretrained.return_value = MagicMock()
            MockTok.from_pretrained.return_value = tok
            m = HFCausalLMModel()
            m.build({"pretrained": "gpt2"})
    assert m.tokenizer is tok


def test_io_type():
    from adapters.llm.models.hf_causal_lm import HFCausalLMModel
    from core.base.io_types import TEXT_TO_TEXT
    assert HFCausalLMModel.io_type == TEXT_TO_TEXT
