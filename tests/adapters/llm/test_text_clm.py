import pytest
import torch
from unittest.mock import MagicMock, patch
from core.base.io_types import TEXT_TO_TEXT


def _make_mock_tokenizer(vocab_size: int = 100, pad_id: int = 0) -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = pad_id
    tok.eos_token = "[EOS]"
    tok.pad_token = "[EOS]"

    def fake_call(text, truncation=True, max_length=128, return_tensors=None):
        ids = torch.arange(1, min(len(text.split()) + 1, max_length + 1))
        mask = torch.ones_like(ids)
        result = MagicMock()
        result.__getitem__ = lambda self, key: {"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0)}[key]
        result.keys = lambda: ["input_ids", "attention_mask"]
        return result

    tok.side_effect = fake_call
    return tok


def _make_dataset(texts=None, max_length=8):
    from adapters.llm.datasets.text_clm import TextCLMDataset

    mock_tok = _make_mock_tokenizer()
    with patch("adapters.llm.datasets.text_clm.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = mock_tok
        ds = TextCLMDataset({"pretrained": "distilgpt2", "texts": texts or ["hello world"], "max_length": max_length})
    return ds, mock_tok


def test_text_clm_registers():
    import adapters.llm.datasets.text_clm
    from core.registry import DATASETS
    assert "text_clm" in DATASETS.list_all()


def test_text_clm_io_type():
    import adapters.llm.datasets.text_clm
    from core.registry import DATASETS
    assert DATASETS.get("text_clm").io_type == TEXT_TO_TEXT


def test_len_matches_texts():
    ds, _ = _make_dataset(["a", "b", "c"])
    assert len(ds) == 3


def test_getitem_returns_required_keys():
    ds, mock_tok = _make_dataset(["hello world"])
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample


def test_getitem_labels_are_clone_of_input_ids():
    ds, mock_tok = _make_dataset(["hello world"])
    sample = ds[0]
    assert torch.equal(sample["input_ids"], sample["labels"])
    # modifying labels should not affect input_ids
    sample["labels"][0] = -999
    assert sample["input_ids"][0] != -999


def test_collate_pads_to_max_len():
    ds, mock_tok = _make_dataset(["a b", "a b c d"])
    # Manually create samples with different lengths
    short = {
        "input_ids": torch.tensor([1, 2]),
        "attention_mask": torch.tensor([1, 1]),
        "labels": torch.tensor([1, 2]),
    }
    long_ = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.tensor([1, 1, 1, 1]),
        "labels": torch.tensor([1, 2, 3, 4]),
    }
    batch = ds.collate_fn([short, long_])
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)


def test_collate_labels_pad_with_minus100():
    ds, mock_tok = _make_dataset(["x"])
    short = {
        "input_ids": torch.tensor([1, 2]),
        "attention_mask": torch.tensor([1, 1]),
        "labels": torch.tensor([1, 2]),
    }
    long_ = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.tensor([1, 1, 1, 1]),
        "labels": torch.tensor([1, 2, 3, 4]),
    }
    batch = ds.collate_fn([short, long_])
    # short sample padded — positions 2 and 3 should be -100 in labels
    assert batch["labels"][0, 2].item() == -100
    assert batch["labels"][0, 3].item() == -100


def test_collate_attention_mask_pad_with_zero():
    ds, mock_tok = _make_dataset(["x"])
    short = {
        "input_ids": torch.tensor([1, 2]),
        "attention_mask": torch.tensor([1, 1]),
        "labels": torch.tensor([1, 2]),
    }
    long_ = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.tensor([1, 1, 1, 1]),
        "labels": torch.tensor([1, 2, 3, 4]),
    }
    batch = ds.collate_fn([short, long_])
    assert batch["attention_mask"][0, 2].item() == 0
    assert batch["attention_mask"][0, 3].item() == 0


def test_tokenizer_from_pretrained_called_with_param():
    from adapters.llm.datasets.text_clm import TextCLMDataset

    mock_tok = _make_mock_tokenizer()
    with patch("adapters.llm.datasets.text_clm.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = mock_tok
        TextCLMDataset({"pretrained": "gpt2", "texts": ["hi"]})
        MockTok.from_pretrained.assert_called_once_with("gpt2")
