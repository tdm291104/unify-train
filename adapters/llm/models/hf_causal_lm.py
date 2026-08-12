from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.base.model import BaseModel
from core.base.io_types import TEXT_TO_TEXT
from core.registry import MODELS


@MODELS.register("hf_causal_lm")
class HFCausalLMModel(BaseModel):

    io_type = TEXT_TO_TEXT

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def build(self, config: dict[str, Any]) -> None:
        name = config.get("pretrained", "gpt2")
        quantization_config = None
        if config.get("load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, config.get("bnb_4bit_compute_dtype", "bfloat16")),
                bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
            )
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quantization_config)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        outputs = self._model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    def save(self, path: str) -> None:
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self._model = AutoModelForCausalLM.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def raw_model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer
