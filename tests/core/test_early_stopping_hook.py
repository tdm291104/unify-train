import pytest
from core.hook.hooks import HookContext
from core.hook.early_stopping import EarlyStoppingHook, StopTraining


def _ctx(loss: float, epoch: int = 0) -> HookContext:
    return HookContext(epoch=epoch, loss=loss)


def test_no_stop_while_improving():
    hook = EarlyStoppingHook(patience=3)
    for loss in [1.0, 0.9, 0.8, 0.7]:
        hook(_ctx(loss))  # should not raise


def test_stops_after_patience_epochs():
    hook = EarlyStoppingHook(patience=2)
    hook(_ctx(1.0))
    hook(_ctx(1.0))  # wait=1
    with pytest.raises(StopTraining):
        hook(_ctx(1.0))  # wait=2 >= patience


def test_improvement_resets_counter():
    hook = EarlyStoppingHook(patience=3)
    hook(_ctx(1.0))   # best=1.0, wait=0
    hook(_ctx(1.1))   # wait=1
    hook(_ctx(0.5))   # improvement, best=0.5, wait=0
    hook(_ctx(0.6))   # wait=1
    hook(_ctx(0.7))   # wait=2
    with pytest.raises(StopTraining):
        hook(_ctx(0.8))  # wait=3 >= patience


def test_min_delta_threshold():
    hook = EarlyStoppingHook(patience=2, min_delta=0.1)
    hook(_ctx(1.0))    # best=1.0
    hook(_ctx(0.95))   # 0.95 >= 1.0 - 0.1 = 0.9 → not improvement, wait=1
    with pytest.raises(StopTraining):
        hook(_ctx(0.95))  # wait=2 >= patience


def test_min_delta_allows_sufficient_improvement():
    hook = EarlyStoppingHook(patience=2, min_delta=0.1)
    hook(_ctx(1.0))    # best=1.0
    hook(_ctx(0.85))   # 0.85 < 1.0 - 0.1 = 0.9 → improvement, wait=0
    hook(_ctx(0.85))   # wait=1
    # not yet raised


def test_invalid_patience_raises():
    with pytest.raises(ValueError, match="patience must be >= 1"):
        EarlyStoppingHook(patience=0)


def test_invalid_min_delta_raises():
    with pytest.raises(ValueError, match="min_delta must be >= 0"):
        EarlyStoppingHook(patience=1, min_delta=-0.1)


def test_stop_training_is_exception():
    assert issubclass(StopTraining, Exception)
