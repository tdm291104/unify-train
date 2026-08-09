import csv
import os
from core.hook.hooks import HookContext


class CSVLoggerHook:
    """Appends a row to a CSV file after every training step."""

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._initialized = False

    def __call__(self, ctx: HookContext) -> None:
        row = {"epoch": ctx.epoch, "step": ctx.step, "loss": ctx.loss, **ctx.metrics}
        if not self._initialized:
            os.makedirs(os.path.dirname(self._filepath) or ".", exist_ok=True)
            with open(self._filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            self._initialized = True
        with open(self._filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
