from abc import ABC, abstractmethod
from typing import Any, ClassVar
from core.base.io_types import IOType


class BaseEvaluator(ABC):

    io_type: ClassVar[IOType]

    @abstractmethod
    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> dict[str, float]:
        """Compute metrics. Returns dict of metric_name -> value."""
