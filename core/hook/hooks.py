from dataclasses import dataclass, field
from typing import Any, Callable

VALID_EVENTS = frozenset(["before_train", "after_train", "before_epoch", "after_epoch", "after_step", "after_eval"])


@dataclass
class HookContext:
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    model: Any = None
    config: Any = None


class HookManager:

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[[HookContext], None]]] = {
            e: [] for e in VALID_EVENTS
        }

    def register(self, event: str, fn: Callable[[HookContext], None]) -> None:
        if event not in VALID_EVENTS:
            raise ValueError(f"Unknown hook event: '{event}'. Valid: {sorted(VALID_EVENTS)}")
        self._hooks[event].append(fn)

    def fire(self, event: str, ctx: HookContext) -> None:
        if event not in VALID_EVENTS:
            raise ValueError(f"Unknown hook event: '{event}'. Valid: {sorted(VALID_EVENTS)}")
        for fn in self._hooks[event]:
            fn(ctx)
