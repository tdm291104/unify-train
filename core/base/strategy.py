from abc import ABC, abstractmethod
from typing import Any, ClassVar
from core.base.io_types import IOType
from core.base.model import BaseModel


class BaseStrategy(ABC):

    compatible_io_types: ClassVar[list[IOType]] = []
    compatible_tasks: ClassVar[list[str]] = []

    def validate(self, io_type: IOType, task: str | None) -> None:
        """Validate before training. Raises ValueError on mismatch."""
        if self.compatible_io_types and io_type not in self.compatible_io_types:
            raise ValueError(
                f"{type(self).__name__} does not support io_type={io_type}. "
                f"Supported: {self.compatible_io_types}"
            )
        if task is not None and self.compatible_tasks and task not in self.compatible_tasks:
            raise ValueError(
                f"{type(self).__name__} does not support task='{task}'. "
                f"Supported: {self.compatible_tasks}"
            )

    @abstractmethod
    def setup(self, model: BaseModel, config: dict[str, Any]) -> BaseModel:
        """Apply strategy to model (e.g., wrap with LoRA). Must return model."""

    @abstractmethod
    def configure_optimizers(self, model: BaseModel, config: dict[str, Any]) -> Any:
        """Create and return the optimizer appropriate for this strategy."""

    def configure_scheduler(self, optimizer: Any, config: dict[str, Any]) -> Any | None:
        """Optionally return an LR scheduler stepped after each epoch. Default: None."""
        return None

    @abstractmethod
    def training_step(
        self,
        model: BaseModel,
        batch: dict[str, Any],
        optimizer: Any,
        step: int,
    ) -> dict[str, float]:
        """Single optimization step. Must return dict with 'loss': float."""

    @abstractmethod
    def teardown(self, model: BaseModel) -> None:
        """Post-training cleanup (e.g., merge LoRA adapters)."""
