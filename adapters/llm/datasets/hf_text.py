from typing import Any
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from core.base.dataset import BaseDataset
from core.base.io_types import TEXT_TO_TEXT
from core.registry import DATASETS


@DATASETS.register("hf_text")
class HFTextDataset(BaseDataset):

    io_type = TEXT_TO_TEXT

    def __init__(self, params: dict[str, Any]) -> None:
        dataset_name: str = params["dataset_name"]
        dataset_config: str | None = params.get("dataset_config")
        split: str = params.get("split", "train")
        self._text_column: str = params.get("text_column", "text")
        self._max_length: int = params.get("max_length", 128)
        filter_empty: bool = params.get("filter_empty", True)

        self._data = load_dataset(dataset_name, dataset_config, split=split)
        if filter_empty:
            col = self._text_column
            self._data = self._data.filter(lambda x: len(x[col].strip()) > 0)

        pretrained: str = params.get("pretrained", "distilgpt2")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text: str = self._data[idx][self._text_column]
        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(s["input_ids"].size(0) for s in samples)
        pad_id = self._tokenizer.pad_token_id
        input_ids_list, attention_mask_list, labels_list = [], [], []
        for s in samples:
            pad = max_len - s["input_ids"].size(0)
            input_ids_list.append(torch.nn.functional.pad(s["input_ids"], (0, pad), value=pad_id))
            attention_mask_list.append(torch.nn.functional.pad(s["attention_mask"], (0, pad), value=0))
            labels_list.append(torch.nn.functional.pad(s["labels"], (0, pad), value=-100))
        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
