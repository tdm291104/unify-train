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
        """Return an LR scheduler stepped after each epoch. Supports 'cosine' and 'step' out of the box."""
        name = config.get("scheduler")
        if name == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=int(config.get("t_max", 3)))
        if name == "step":
            from torch.optim.lr_scheduler import StepLR
            return StepLR(
                optimizer,
                step_size=int(config.get("step_size", 1)),
                gamma=float(config.get("gamma", 0.1)),
            )
        return None

    @abstractmethod
    def training_step(
        self,
        model: BaseModel,
        batch: dict[str, Any],
        step: int,
    ) -> dict[str, Any]:
        """Forward pass only. Must return dict with 'loss': Tensor (no backward, no optimizer calls)."""

    @abstractmethod
    def teardown(self, model: BaseModel) -> None:
        """Post-training cleanup (e.g., merge LoRA adapters)."""
