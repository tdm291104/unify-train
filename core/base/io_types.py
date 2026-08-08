from dataclasses import dataclass
from enum import Enum


class DataType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    CATEGORY = "category"
    TABULAR = "tabular"


@dataclass(frozen=True)
class IOType:
    input: DataType
    output: DataType

    def __str__(self) -> str:
        return f"{self.input.value}->{self.output.value}"


TEXT_TO_TEXT = IOType(DataType.TEXT, DataType.TEXT)
IMAGE_TO_CATEGORY = IOType(DataType.IMAGE, DataType.CATEGORY)
AUDIO_TO_TEXT = IOType(DataType.AUDIO, DataType.TEXT)
