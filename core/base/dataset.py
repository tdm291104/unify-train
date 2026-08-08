from abc import ABC, abstractmethod
from typing import Any, ClassVar
from core.base.io_types import IOType


class BaseDataset(ABC):

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
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    @abstractmethod
    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into a batch dict for DataLoader."""
