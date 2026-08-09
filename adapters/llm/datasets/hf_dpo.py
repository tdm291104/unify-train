from typing import Any
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from core.base.dataset import BaseDataset
from core.base.io_types import TEXT_TO_TEXT
from core.registry import DATASETS


@DATASETS.register("hf_dpo")
class HFDPODataset(BaseDataset):
    """Dataset for DPO training. Expects rows with `chosen` and `rejected` text fields."""

    io_type = TEXT_TO_TEXT

    def __init__(self, params: dict[str, Any]) -> None:
        dataset_name: str = params["dataset_name"]
        dataset_config: str | None = params.get("dataset_config")
        split: str = params.get("split", "train")
        self._chosen_column: str = params.get("chosen_column", "chosen")
        self._rejected_column: str = params.get("rejected_column", "rejected")
        self._max_length: int = params.get("max_length", 128)

        self._data = load_dataset(dataset_name, dataset_config, split=split)

        pretrained: str = params.get("pretrained", "distilgpt2")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def __len__(self) -> int:
        return len(self._data)

    def _encode(self, text: str) -> dict[str, torch.Tensor]:
        enc = self._tokenizer(
            text,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._data[idx]
        chosen = self._encode(row[self._chosen_column])
        rejected = self._encode(row[self._rejected_column])
        return {"chosen": chosen, "rejected": rejected}

    def _pad_sequence(self, seq: dict[str, torch.Tensor], target_len: int) -> dict[str, torch.Tensor]:
        pad_id = self._tokenizer.pad_token_id
        pad = target_len - seq["input_ids"].size(0)
        return {
            "input_ids": torch.nn.functional.pad(seq["input_ids"], (0, pad), value=pad_id),
            "attention_mask": torch.nn.functional.pad(seq["attention_mask"], (0, pad), value=0),
            "labels": torch.nn.functional.pad(seq["labels"], (0, pad), value=-100),
        }

    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        chosen_max = max(s["chosen"]["input_ids"].size(0) for s in samples)
        rejected_max = max(s["rejected"]["input_ids"].size(0) for s in samples)

        chosen_list = [self._pad_sequence(s["chosen"], chosen_max) for s in samples]
        rejected_list = [self._pad_sequence(s["rejected"], rejected_max) for s in samples]

        def _stack(seq_list: list[dict]) -> dict[str, torch.Tensor]:
            return {k: torch.stack([s[k] for s in seq_list]) for k in seq_list[0]}

        return {"chosen": _stack(chosen_list), "rejected": _stack(rejected_list)}
