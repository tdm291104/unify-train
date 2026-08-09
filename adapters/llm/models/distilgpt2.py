from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.base.model import BaseModel
from core.base.io_types import TEXT_TO_TEXT
from core.registry import MODELS


@MODELS.register("distilgpt2")
class DistilGPT2Model(BaseModel):

    io_type = TEXT_TO_TEXT

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def build(self, config: dict[str, Any]) -> None:
        name = config.get("pretrained", "distilgpt2")
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(name)

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
    def raw_model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer
