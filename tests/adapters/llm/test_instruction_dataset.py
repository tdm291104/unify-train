import json
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch


def _mock_tokenizer(vocab_size=100):
    tok = MagicMock()
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0

    def tokenize(text, truncation=True, max_length=512, add_special_tokens=False, **kwargs):
        ids = [i % vocab_size for i in range(min(len(text.split()), max_length))]
        return {"input_ids": ids}

    tok.side_effect = tokenize
    return tok


def _make_alpaca_file(tmp_path, n=3):
    path = tmp_path / "data.jsonl"
    rows = [
        {"instruction": f"Do task {i}", "input": "", "output": f"Result {i}"}
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def _make_sharegpt_file(tmp_path, n=3):
    path = tmp_path / "sharegpt.jsonl"
    rows = [
        {"conversations": [{"from": "human", "value": f"Q{i}"}, {"from": "gpt", "value": f"A{i}"}]}
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def test_registers():
    import adapters.llm.datasets.instruction  # noqa: F401
    from core.registry import DATASETS
    assert "instruction" in DATASETS.list_all()


def test_alpaca_length(tmp_path):
    from adapters.llm.datasets.instruction import InstructionDataset

    path = _make_alpaca_file(tmp_path, n=5)
    with patch("adapters.llm.datasets.instruction.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = _mock_tokenizer()
        ds = InstructionDataset({"path": path, "pretrained": "gpt2", "format": "alpaca"})
    assert len(ds) == 5


def test_sharegpt_length(tmp_path):
    from adapters.llm.datasets.instruction import InstructionDataset

    path = _make_sharegpt_file(tmp_path, n=4)
    with patch("adapters.llm.datasets.instruction.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = _mock_tokenizer()
        ds = InstructionDataset({"path": path, "pretrained": "gpt2", "format": "sharegpt"})
    assert len(ds) == 4


def test_auto_detect_alpaca(tmp_path):
    from adapters.llm.datasets.instruction import InstructionDataset

    path = _make_alpaca_file(tmp_path)
    with patch("adapters.llm.datasets.instruction.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = _mock_tokenizer()
        ds = InstructionDataset({"path": path, "pretrained": "gpt2"})
    assert ds._format == "alpaca"


def test_auto_detect_sharegpt(tmp_path):
    from adapters.llm.datasets.instruction import InstructionDataset

    path = _make_sharegpt_file(tmp_path)
    with patch("adapters.llm.datasets.instruction.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = _mock_tokenizer()
        ds = InstructionDataset({"path": path, "pretrained": "gpt2"})
    assert ds._format == "sharegpt"


def test_unknown_format_raises(tmp_path):
    from adapters.llm.datasets.instruction import InstructionDataset

    path = _make_alpaca_file(tmp_path)
    with patch("adapters.llm.datasets.instruction.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = _mock_tokenizer()
        with pytest.raises(ValueError, match="Unknown format"):
            InstructionDataset({"path": path, "pretrained": "gpt2", "format": "csv"})


def test_getitem_returns_required_keys(tmp_path):
    from adapters.llm.datasets.instruction import InstructionDataset

    path = _make_alpaca_file(tmp_path)
    with patch("adapters.llm.datasets.instruction.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = _mock_tokenizer()
        ds = InstructionDataset({"path": path, "pretrained": "gpt2", "format": "alpaca", "max_length": 32})
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)


def test_prompt_tokens_masked_in_labels(tmp_path):
    from adapters.llm.datasets.instruction import InstructionDataset

    path = _make_alpaca_file(tmp_path)

    tok = MagicMock()
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0

    call_count = [0]

    def tokenize(text, truncation=True, max_length=512, add_special_tokens=False, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"input_ids": [10, 11, 12]}   # prompt: 3 tokens
        else:
            return {"input_ids": [20, 21]}        # response: 2 tokens

    tok.side_effect = tokenize

    with patch("adapters.llm.datasets.instruction.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = tok
        ds = InstructionDataset({"path": path, "pretrained": "gpt2", "format": "alpaca"})

    item = ds[0]
    labels = item["labels"].tolist()
    # First 3 tokens (prompt) must be -100
    assert labels[:3] == [-100, -100, -100]
    # Last 2 tokens (response) must not be -100
    assert -100 not in labels[3:]


def test_collate_fn_pads_to_max(tmp_path):
    from adapters.llm.datasets.instruction import InstructionDataset

    path = _make_alpaca_file(tmp_path)
    with patch("adapters.llm.datasets.instruction.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = _mock_tokenizer()
        ds = InstructionDataset({"path": path, "pretrained": "gpt2", "format": "alpaca", "max_length": 32})

    s1 = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "labels": torch.tensor([-100, 2, 3]),
    }
    s2 = {
        "input_ids": torch.tensor([4, 5]),
        "attention_mask": torch.tensor([1, 1]),
        "labels": torch.tensor([-100, 5]),
    }
    batch = ds.collate_fn([s1, s2])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    # Shorter sequence padded with 0 for input_ids and -100 for labels
    assert batch["input_ids"][1, 2].item() == 0
    assert batch["labels"][1, 2].item() == -100
