from typing import Any
from pydantic import BaseModel as PydanticBase, field_validator
from core.base.io_types import DataType


class ModelConfig(PydanticBase):
    name: str
    params: dict[str, Any] = {}


class DatasetConfig(PydanticBase):
    name: str
    params: dict[str, Any] = {}


class StrategyConfig(PydanticBase):
    name: str
    params: dict[str, Any] = {}


class EvaluatorConfig(PydanticBase):
    name: str


class IOTypeConfig(PydanticBase):
    input: str
    output: str

    @field_validator("input", "output")
    @classmethod
    def must_be_valid_data_type(cls, v: str) -> str:
        DataType(v)  # raises ValueError if invalid
        return v


class TrainConfig(PydanticBase):
    max_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    log_every_n_steps: int = 10


class UnifyTrainConfig(PydanticBase):
    model: ModelConfig
    dataset: DatasetConfig
    strategy: StrategyConfig
    evaluator: EvaluatorConfig
    io_type: IOTypeConfig
    task: str | None = None
    train: TrainConfig = TrainConfig()
    output_dir: str = "outputs/"
