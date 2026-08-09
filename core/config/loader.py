import yaml
from pathlib import Path
from core.config.schema import UnifyTrainConfig


def load_config(path: str) -> UnifyTrainConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}: {path}")
    return UnifyTrainConfig(**raw)
