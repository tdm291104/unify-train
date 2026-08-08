import yaml
from pathlib import Path
from core.config.schema import UnifyTrainConfig


def load_config(path: str) -> UnifyTrainConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open() as f:
        raw = yaml.safe_load(f)
    return UnifyTrainConfig(**raw)
