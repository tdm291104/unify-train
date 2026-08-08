from abc import ABC, abstractmethod
from typing import Any, ClassVar
from core.base.io_types import IOType


class BaseDataset(ABC):

    io_type: ClassVar[IOType]

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    @abstractmethod
    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into a batch dict for DataLoader."""
