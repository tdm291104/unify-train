import pytest
import torch
from unittest.mock import MagicMock, patch


def _make_dataset(rows=None):
    """Build HFDPODataset with mocked HF dataset and tokenizer."""
    if rows is None:
        rows = [
            {"chosen": "I love AI", "rejected": "I hate AI"},
            {"chosen": "Great day", "rejected": "Bad day"},
        ]
    mock_ds = rows

    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "<|endoftext|>"
    mock_tok.pad_token_id = 50256

    def tokenize(text, truncation=True, max_length=128, return_tensors="pt"):
        length = min(len(text.split()), max_length)
        ids = torch.ones(1, length, dtype=torch.long)
        mask = torch.ones(1, length, dtype=torch.long)
        result = MagicMock()
        result.__getitem__ = lambda self, k: ids if k == "input_ids" else mask
        result.get = lambda k, d=None: ids if k == "input_ids" else mask
        return {"input_ids": ids, "attention_mask": mask}

    mock_tok.side_effect = tokenize

    with patch("adapters.llm.datasets.hf_dpo.load_dataset", return_value=mock_ds), \
         patch("adapters.llm.datasets.hf_dpo.AutoTokenizer.from_pretrained", return_value=mock_tok):
        from adapters.llm.datasets.hf_dpo import HFDPODataset
        return HFDPODataset({
            "dataset_name": "dummy",
            "pretrained": "distilgpt2",
        })


def test_hf_dpo_registers():
    import adapters.llm.datasets.hf_dpo
    from core.registry import DATASETS
    assert "hf_dpo" in DATASETS.list_all()


def test_hf_dpo_io_type():
    from adapters.llm.datasets.hf_dpo import HFDPODataset
    from core.base.io_types import TEXT_TO_TEXT
    assert HFDPODataset.io_type == TEXT_TO_TEXT


def test_hf_dpo_len():
    ds = _make_dataset()
    assert len(ds) == 2


def test_hf_dpo_getitem_has_chosen_and_rejected():
    ds = _make_dataset()
    item = ds[0]
    assert "chosen" in item
    assert "rejected" in item


def test_hf_dpo_getitem_chosen_has_tensors():
    ds = _make_dataset()
    item = ds[0]
    assert "input_ids" in item["chosen"]
    assert "attention_mask" in item["chosen"]
    assert "labels" in item["chosen"]


def test_hf_dpo_collate_fn_stacks():
    ds = _make_dataset()
    items = [ds[0], ds[1]]
    batch = ds.collate_fn(items)
    assert "chosen" in batch
    assert "rejected" in batch
    assert batch["chosen"]["input_ids"].dim() == 2
    assert batch["rejected"]["input_ids"].dim() == 2


def test_hf_dpo_collate_batch_size():
    ds = _make_dataset()
    items = [ds[0], ds[1]]
    batch = ds.collate_fn(items)
    assert batch["chosen"]["input_ids"].shape[0] == 2
    assert batch["rejected"]["input_ids"].shape[0] == 2
