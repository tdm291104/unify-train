from typing import Any
import torch
from transformers import AutoTokenizer
from core.base.dataset import BaseDataset
from core.base.io_types import TEXT_TO_TEXT
from core.registry import DATASETS


@DATASETS.register("text_clm")
class TextCLMDataset(BaseDataset):

    io_type = TEXT_TO_TEXT

    def __init__(self, params: dict[str, Any]) -> None:
        pretrained = params.get("pretrained", "distilgpt2")
        self._texts: list[str] = params.get("texts", [])
        self._max_length: int = params.get("max_length", 128)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        encoded = self._tokenizer(
            self._texts[idx],
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
        input_ids_list, attention_mask_list, labels_list = [], [], []
        pad_id = self._tokenizer.pad_token_id
        for s in samples:
            seq_len = s["input_ids"].size(0)
            pad = max_len - seq_len
            input_ids_list.append(torch.nn.functional.pad(s["input_ids"], (0, pad), value=pad_id))
            attention_mask_list.append(torch.nn.functional.pad(s["attention_mask"], (0, pad), value=0))
            labels_list.append(torch.nn.functional.pad(s["labels"], (0, pad), value=-100))
        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
