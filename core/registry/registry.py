from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str):
        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise KeyError(
                    f"'{name}' already registered in Registry('{self._name}'). "
                    f"Existing: {type(self._registry[name]).__name__}"
                )
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> type[T]:
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in Registry('{self._name}'). "
                f"Available: {self.list_all()}"
            )
        return self._registry[name]

    def list_all(self) -> list[str]:
        return list(self._registry.keys())


MODELS: Registry = Registry("models")
DATASETS: Registry = Registry("datasets")
STRATEGIES: Registry = Registry("strategies")
EVALUATORS: Registry = Registry("evaluators")
