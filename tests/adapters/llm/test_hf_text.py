import pytest
import torch
from unittest.mock import MagicMock, patch
from core.base.io_types import TEXT_TO_TEXT


def _make_mock_hf_dataset(texts: list[str]):
    """Returns a mock that behaves like a HuggingFace Dataset."""
    ds = MagicMock()
    rows = [{"text": t} for t in texts]
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    ds.filter = MagicMock(return_value=ds)
    return ds


def _make_mock_tokenizer(pad_id: int = 0):
    tok = MagicMock()
    tok.pad_token_id = pad_id
    tok.eos_token = "[EOS]"
    tok.pad_token = "[EOS]"

    def fake_call(text, truncation=True, max_length=128, return_tensors=None):
        ids = torch.arange(1, min(len(text.split()) + 1, max_length + 1))
        mask = torch.ones_like(ids)
        result = MagicMock()
        result.__getitem__ = lambda self, k: {"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0)}[k]
        return result

    tok.side_effect = fake_call
    return tok


def _make_dataset(texts=None, extra_params=None):
    from adapters.llm.datasets.hf_text import HFTextDataset
    mock_hf = _make_mock_hf_dataset(texts or ["hello world", "foo bar baz"])
    mock_tok = _make_mock_tokenizer()
    params = {"dataset_name": "wikitext", "pretrained": "distilgpt2", **(extra_params or {})}
    with patch("adapters.llm.datasets.hf_text.load_dataset", return_value=mock_hf):
        with patch("adapters.llm.datasets.hf_text.AutoTokenizer") as MockTok:
            MockTok.from_pretrained.return_value = mock_tok
            ds = HFTextDataset(params)
    return ds, mock_hf, mock_tok


def test_hf_text_registers():
    import adapters.llm.datasets.hf_text
    from core.registry import DATASETS
    assert "hf_text" in DATASETS.list_all()


def test_hf_text_io_type():
    import adapters.llm.datasets.hf_text
    from core.registry import DATASETS
    assert DATASETS.get("hf_text").io_type == TEXT_TO_TEXT


def test_len_matches_dataset():
    ds, mock_hf, _ = _make_dataset(["a", "b", "c"])
    assert len(ds) == 3


def test_getitem_returns_required_keys():
    ds, _, _ = _make_dataset(["hello world"])
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample


def test_getitem_labels_clone_of_input_ids():
    ds, _, _ = _make_dataset(["hello world"])
    sample = ds[0]
    assert torch.equal(sample["input_ids"], sample["labels"])
    sample["labels"][0] = -999
    assert sample["input_ids"][0] != -999


def test_filter_empty_called_by_default():
    ds, mock_hf, _ = _make_dataset(["hello"])
    mock_hf.filter.assert_called_once()


def test_filter_empty_skipped_when_disabled():
    ds, mock_hf, _ = _make_dataset(["hello"], extra_params={"filter_empty": False})
    mock_hf.filter.assert_not_called()


def test_load_dataset_called_with_correct_args():
    from adapters.llm.datasets.hf_text import HFTextDataset
    mock_hf = _make_mock_hf_dataset(["hi"])
    mock_tok = _make_mock_tokenizer()
    params = {
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "validation",
        "pretrained": "distilgpt2",
    }
    with patch("adapters.llm.datasets.hf_text.load_dataset", return_value=mock_hf) as mock_ld:
        with patch("adapters.llm.datasets.hf_text.AutoTokenizer") as MockTok:
            MockTok.from_pretrained.return_value = mock_tok
            HFTextDataset(params)
    mock_ld.assert_called_once_with("wikitext", "wikitext-2-raw-v1", split="validation")


def test_collate_pads_to_max_len():
    ds, _, _ = _make_dataset(["a b", "x y z w"])
    short = {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]), "labels": torch.tensor([1, 2])}
    long_ = {"input_ids": torch.tensor([1, 2, 3, 4]), "attention_mask": torch.tensor([1, 1, 1, 1]), "labels": torch.tensor([1, 2, 3, 4])}
    batch = ds.collate_fn([short, long_])
    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"][0, 2].item() == -100
    assert batch["attention_mask"][0, 2].item() == 0


def test_text_column_configurable():
    from adapters.llm.datasets.hf_text import HFTextDataset
    rows = [{"content": "hello world"}]
    mock_hf = MagicMock()
    mock_hf.__len__ = MagicMock(return_value=1)
    mock_hf.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    mock_hf.filter = MagicMock(return_value=mock_hf)
    mock_tok = _make_mock_tokenizer()
    params = {"dataset_name": "custom", "text_column": "content", "pretrained": "distilgpt2"}
    with patch("adapters.llm.datasets.hf_text.load_dataset", return_value=mock_hf):
        with patch("adapters.llm.datasets.hf_text.AutoTokenizer") as MockTok:
            MockTok.from_pretrained.return_value = mock_tok
            ds = HFTextDataset(params)
    sample = ds[0]
    assert "input_ids" in sample
