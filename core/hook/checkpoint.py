import os
from core.hook.hooks import HookContext


class CheckpointHook:
    """Saves the model after every `every` epochs."""

    def __init__(self, checkpoint_dir: str, every: int = 1) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self._dir = checkpoint_dir
        self._every = every

    def __call__(self, ctx: HookContext) -> None:
        if (ctx.epoch + 1) % self._every != 0:
            return
        epoch_dir = os.path.join(self._dir, f"epoch_{ctx.epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)
        ctx.model.save(epoch_dir)
