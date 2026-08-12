import sys
from unittest.mock import MagicMock, patch

import pytest

from core.hook.hooks import HookContext
from core.hook.wandb_logger import WandbLoggerHook


def _make_ctx(**kwargs) -> HookContext:
    defaults = {"epoch": 1, "step": 10, "loss": 0.5, "metrics": {}}
    defaults.update(kwargs)
    return HookContext(**defaults)


def test_on_before_train_calls_wandb_init():
    mock_wandb = MagicMock()
    hook = WandbLoggerHook(
        project="my-project",
        run_name="run-1",
        config={"lr": 1e-4},
        tags=["test"],
    )
    ctx = _make_ctx()
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        hook.on_before_train(ctx)
        mock_wandb.init.assert_called_once_with(
            project="my-project",
            name="run-1",
            config={"lr": 1e-4},
            tags=["test"],
        )


def test_on_after_step_logs_loss():
    mock_wandb = MagicMock()
    hook = WandbLoggerHook(project="my-project")
    ctx = _make_ctx(loss=0.5, step=10, metrics={})
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        hook.on_before_train(ctx)
        hook.on_after_step(ctx)
        mock_wandb.log.assert_called_once_with(
            {"train/loss": 0.5},
            step=10,
        )


def test_on_after_step_logs_extra_metrics():
    mock_wandb = MagicMock()
    hook = WandbLoggerHook(project="my-project")
    ctx = _make_ctx(loss=0.5, step=5, metrics={"loss": 0.5, "sft_loss": 0.3})
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        hook.on_before_train(ctx)
        hook.on_after_step(ctx)
        call_args = mock_wandb.log.call_args
        payload = call_args[0][0]
        # train/loss comes from ctx.loss, not metrics["loss"]
        assert payload["train/loss"] == 0.5
        # sft_loss gets the train/ prefix
        assert payload["train/sft_loss"] == 0.3
        # "loss" key from metrics must NOT appear as its own entry (not double-logged)
        assert "train/loss" in payload
        # Only one train/loss key
        assert len([k for k in payload if k == "train/loss"]) == 1
        # metrics["loss"] is not separately logged as a non-prefixed key
        assert "loss" not in payload


def test_on_after_eval_logs_eval_metrics():
    mock_wandb = MagicMock()
    hook = WandbLoggerHook(project="my-project")
    ctx = _make_ctx(step=20, metrics={"bleu": 45.2})
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        hook.on_before_train(ctx)
        hook.on_after_eval(ctx)
        mock_wandb.log.assert_called_once_with(
            {"eval/bleu": 45.2},
            step=20,
        )


def test_on_after_train_calls_finish():
    mock_wandb = MagicMock()
    hook = WandbLoggerHook(project="my-project")
    ctx = _make_ctx()
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        hook.on_before_train(ctx)
        hook.on_after_train(ctx)
        mock_wandb.finish.assert_called_once()


def test_on_after_step_before_init_is_noop():
    hook = WandbLoggerHook(project="my-project")
    ctx = _make_ctx()
    # Should not raise even though on_before_train was never called
    hook.on_after_step(ctx)


def test_missing_wandb_raises_import_error():
    hook = WandbLoggerHook(project="my-project")
    ctx = _make_ctx()
    with patch.dict(sys.modules, {"wandb": None}):
        with pytest.raises(ImportError, match="wandb"):
            hook.on_before_train(ctx)
