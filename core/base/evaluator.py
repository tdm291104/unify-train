from abc import ABC, abstractmethod
from typing import Any, ClassVar
from core.base.io_types import IOType


class BaseEvaluator(ABC):

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
    def extract(
        self,
        model_output: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple[list[Any], list[Any]]:
        """Extract (predictions, references) from a single batch's model output."""

    @abstractmethod
    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> dict[str, float]:
        """Compute metrics. Returns dict of metric_name -> value."""
