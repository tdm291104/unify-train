import os
import pytest
from unittest.mock import MagicMock
from core.hook.hooks import HookContext
from core.hook.checkpoint import CheckpointHook


def _ctx(epoch: int, model=None) -> HookContext:
    ctx = HookContext(epoch=epoch)
    ctx.model = model or MagicMock()
    return ctx


def test_saves_on_first_epoch(tmp_path):
    hook = CheckpointHook(str(tmp_path), every=1)
    model = MagicMock()
    hook(_ctx(epoch=0, model=model))
    model.save.assert_called_once_with(str(tmp_path / "epoch_1"))


def test_saves_every_n_epochs(tmp_path):
    hook = CheckpointHook(str(tmp_path), every=2)
    model = MagicMock()
    for epoch in range(4):
        hook(_ctx(epoch=epoch, model=model))
    assert model.save.call_count == 2
    calls = [c.args[0] for c in model.save.call_args_list]
    assert any("epoch_2" in c for c in calls)
    assert any("epoch_4" in c for c in calls)


def test_skips_non_multiple_epochs(tmp_path):
    hook = CheckpointHook(str(tmp_path), every=3)
    model = MagicMock()
    hook(_ctx(epoch=0, model=model))
    hook(_ctx(epoch=1, model=model))
    model.save.assert_not_called()


def test_creates_epoch_subdirectory(tmp_path):
    hook = CheckpointHook(str(tmp_path), every=1)
    model = MagicMock()
    hook(_ctx(epoch=2, model=model))
    model.save.assert_called_once_with(str(tmp_path / "epoch_3"))


def test_invalid_every_raises():
    with pytest.raises(ValueError, match="every must be >= 1"):
        CheckpointHook("/tmp", every=0)
