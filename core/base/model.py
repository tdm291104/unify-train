from abc import ABC, abstractmethod
from typing import Any, ClassVar
from core.base.io_types import IOType


class BaseModel(ABC):

    io_type: ClassVar[IOType]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # ABCMeta sets cls.__abstractmethods__ after __init_subclass__ runs,
        # so we compute concreteness by walking the MRO ourselves.
        seen: set[str] = set()
        remaining_abstract: set[str] = set()
        for klass in cls.__mro__:
            for name, val in vars(klass).items():
                if name not in seen:
                    seen.add(name)
                    if getattr(val, "__isabstractmethod__", False):
                        remaining_abstract.add(name)
        if not remaining_abstract and not hasattr(cls, "io_type"):
            raise TypeError(
                f"{cls.__name__} must define class attribute 'io_type: ClassVar[IOType]'"
            )

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

    @property
    @abstractmethod
    def raw_model(self) -> Any:
        """The underlying torch.nn.Module."""
