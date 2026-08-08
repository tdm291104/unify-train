import pytest
from core.registry.registry import Registry


def test_register_and_get():
    reg = Registry("test")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo


def test_register_preserves_class_behavior():
    reg = Registry("test")

    @reg.register("myclass")
    class MyClass:
        def hello(self): return "hi"

    assert reg.get("myclass")().hello() == "hi"


def test_register_duplicate_raises():
    reg = Registry("test")

    @reg.register("bar")
    class Bar1:
        pass

    with pytest.raises(KeyError, match="already registered"):
        @reg.register("bar")
        class Bar2:
            pass


def test_get_unknown_raises_with_hint():
    reg = Registry("models")

    @reg.register("known")
    class Known:
        pass

    with pytest.raises(KeyError, match="known"):
        reg.get("unknown")


def test_list_all():
    reg = Registry("test")

    @reg.register("a")
    class A: pass

    @reg.register("b")
    class B: pass

    assert set(reg.list_all()) == {"a", "b"}


def test_empty_registry_list_all():
    assert Registry("empty").list_all() == []


def test_global_singletons_exist():
    from core.registry.registry import MODELS, DATASETS, STRATEGIES, EVALUATORS
    assert MODELS is not None
    assert DATASETS is not None
    assert STRATEGIES is not None
    assert EVALUATORS is not None
