from dataclasses import dataclass
from enum import Enum


class DataType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    CATEGORY = "category"
    TABULAR = "tabular"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class IOType:
    input: DataType
    output: DataType

    def __post_init__(self) -> None:
        if not isinstance(self.input, DataType):
            raise TypeError(f"input must be a DataType, got {type(self.input).__name__!r}")
        if not isinstance(self.output, DataType):
            raise TypeError(f"output must be a DataType, got {type(self.output).__name__!r}")

    def __str__(self) -> str:
        return f"{self.input.value}->{self.output.value}"


TEXT_TO_TEXT = IOType(DataType.TEXT, DataType.TEXT)
IMAGE_TO_CATEGORY = IOType(DataType.IMAGE, DataType.CATEGORY)
AUDIO_TO_TEXT = IOType(DataType.AUDIO, DataType.TEXT)
