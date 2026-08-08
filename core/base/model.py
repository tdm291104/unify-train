from abc import ABC, abstractmethod
from typing import Any, ClassVar
from core.base.io_types import IOType


class BaseModel(ABC):

    io_type: ClassVar[IOType]

    @abstractmethod
    def build(self, config: dict[str, Any]) -> None:
        """Load weights and apply config."""

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Forward pass. Output must contain 'loss': Tensor."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist weights/adapters to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load weights/adapters from disk."""
