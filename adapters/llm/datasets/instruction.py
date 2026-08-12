import json
from typing import Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from core.base.dataset import BaseDataset
from core.base.io_types import TEXT_TO_TEXT
from core.registry import DATASETS

_ALPACA_PROMPT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
_ALPACA_PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"


@DATASETS.register("instruction")
class InstructionDataset(BaseDataset):

    io_type = TEXT_TO_TEXT

    def __init__(self, params: dict[str, Any]) -> None:
        path: str = params["path"]
        pretrained: str = params["pretrained"]
        self._max_length: int = params.get("max_length", 512)
        fmt: str | None = params.get("format")

        if path.endswith(".json") or path.endswith(".jsonl"):
            with open(path) as f:
                self._data = [json.loads(line) for line in f if line.strip()]
        else:
            from datasets import load_dataset
            ds = load_dataset(path, split=params.get("split", "train"))
            self._data = list(ds)

        if fmt is None:
            fmt = "sharegpt" if "conversations" in self._data[0] else "alpaca"

        if fmt not in ("alpaca", "sharegpt"):
            raise ValueError(f"Unknown format '{fmt}'. Supported: alpaca, sharegpt")

        self._format = fmt
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def __len__(self) -> int:
        return len(self._data)

    def _build_alpaca(self, row: dict) -> tuple[str, str]:
        instr = row["instruction"]
        inp = row.get("input", "").strip()
        out = row["output"]
        prompt = (
            _ALPACA_PROMPT.format(instruction=instr, input=inp)
            if inp
            else _ALPACA_PROMPT_NO_INPUT.format(instruction=instr)
        )
        return prompt, out

    def _build_sharegpt(self, row: dict) -> tuple[str, str]:
        convs = row["conversations"]
        human = next((c["value"] for c in convs if c["from"] == "human"), "")
        gpt = next((c["value"] for c in convs if c["from"] == "gpt"), "")
        return f"### Human:\n{human}\n\n### Assistant:\n", gpt

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._data[idx]
        if self._format == "alpaca":
            prompt, response = self._build_alpaca(row)
        else:
            prompt, response = self._build_sharegpt(row)

        prompt_ids = self._tokenizer(
            prompt,
            truncation=True,
            max_length=self._max_length,
            add_special_tokens=False,
        )["input_ids"]
        resp_max = max(1, self._max_length - len(prompt_ids))
        response_ids = self._tokenizer(
            response,
            truncation=True,
            max_length=resp_max,
            add_special_tokens=False,
        )["input_ids"]

        all_ids = prompt_ids + response_ids
        input_ids = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([-100] * len(prompt_ids) + response_ids, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(s["input_ids"].size(0) for s in samples)
        pad_id = self._tokenizer.pad_token_id

        def _pad(seq: torch.Tensor, val: int) -> torch.Tensor:
            return F.pad(seq, (0, max_len - seq.size(0)), value=val)

        return {
            "input_ids": torch.stack([_pad(s["input_ids"], pad_id) for s in samples]),
            "attention_mask": torch.stack([_pad(s["attention_mask"], 0) for s in samples]),
            "labels": torch.stack([_pad(s["labels"], -100) for s in samples]),
        }
