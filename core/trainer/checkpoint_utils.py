import json
import os


def load_checkpoint(path: str, model) -> tuple[int, int]:
    """Load model weights from checkpoint and return (start_epoch, start_step).

    start_epoch is the first epoch to run (epochs before it are skipped).
    start_step is the global step count to continue from.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")
    model.load(path)
    meta_path = os.path.join(path, "checkpoint_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return int(meta.get("epoch", 0)), int(meta.get("step", 0))
    return 0, 0
