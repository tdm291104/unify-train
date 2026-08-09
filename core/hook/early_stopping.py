from core.hook.hooks import HookContext


class StopTraining(Exception):
    """Raised by EarlyStoppingHook to signal the Trainer to stop early."""


class EarlyStoppingHook:
    """Raises StopTraining when monitored loss shows no improvement for `patience` epochs."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        self._patience = patience
        self._min_delta = min_delta
        self._best: float = float("inf")
        self._wait: int = 0

    def __call__(self, ctx: HookContext) -> None:
        if ctx.loss < self._best - self._min_delta:
            self._best = ctx.loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                raise StopTraining(
                    f"No improvement for {self._patience} epochs "
                    f"(best={self._best:.4f}, current={ctx.loss:.4f})"
                )
