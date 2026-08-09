import pytest
import os
import torch
from unittest.mock import MagicMock, patch
from core.trainer.trainer import Trainer, build_trainer_from_config
from core.hook.hooks import HookManager, HookContext
from core.config.schema import UnifyTrainConfig


def _make_config(**overrides):
    base = {
        "model": {"name": "mock_model"},
        "dataset": {"name": "mock_dataset"},
        "strategy": {"name": "mock_strategy"},
        "evaluator": {"name": "mock_evaluator"},
        "io_type": {"input": "text", "output": "text"},
        "train": {"max_epochs": 1, "batch_size": 2, "log_every_n_steps": 1},
        "output_dir": "/tmp/test_out",
    }
    base.update(overrides)
    return UnifyTrainConfig(**base)


def _make_strategy(training_step_return=None):
    s = MagicMock()
    s.setup = MagicMock(side_effect=lambda m, c: m)
    s.configure_optimizers = MagicMock(return_value=MagicMock())
    s.training_step = MagicMock(
        return_value=training_step_return or {"loss": torch.tensor(0.1, requires_grad=True)}
    )
    s.teardown = MagicMock()
    s.validate = MagicMock()
    return s


def test_trainer_calls_validate_before_setup():
    strategy = _make_strategy()
    strategy.validate.side_effect = ValueError("incompatible")

    trainer = Trainer(
        model=MagicMock(),
        dataset=MagicMock(),
        strategy=strategy,
        config=_make_config(),
    )
    with pytest.raises(ValueError, match="incompatible"):
        trainer.run()

    strategy.validate.assert_called_once()
    strategy.setup.assert_not_called()


def test_trainer_validate_receives_correct_io_type():
    from core.base.io_types import TEXT_TO_TEXT
    strategy = _make_strategy()

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = []
        with patch("os.makedirs"):
            trainer = Trainer(
                model=MagicMock(),
                dataset=MagicMock(),
                strategy=strategy,
                config=_make_config(),
            )
            trainer.run()

    args, kwargs = strategy.validate.call_args
    assert args[0] == TEXT_TO_TEXT
    assert args[1] is None  # task=None in config


def test_trainer_calls_lifecycle_in_order():
    order = []
    strategy = MagicMock()
    strategy.validate = MagicMock(side_effect=lambda *a: order.append("validate"))
    strategy.setup = MagicMock(side_effect=lambda m, c: (order.append("setup"), m)[1])
    strategy.configure_optimizers = MagicMock(
        side_effect=lambda m, c: (order.append("configure_optimizers"), MagicMock())[1]
    )
    strategy.training_step = MagicMock(return_value={"loss": torch.tensor(0.1, requires_grad=True)})
    strategy.teardown = MagicMock(side_effect=lambda m: order.append("teardown"))

    hooks = HookManager()
    hooks.register("before_train", lambda ctx: order.append("before_train"))
    hooks.register("after_train", lambda ctx: order.append("after_train"))

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        with patch("os.makedirs"):
            MockDL.return_value = [{"input_ids": [0]}]
            trainer = Trainer(
                model=MagicMock(),
                dataset=MagicMock(),
                strategy=strategy,
                config=_make_config(),
                hooks=hooks,
            )
            trainer.run()

    assert order.index("validate") < order.index("setup")
    assert order.index("setup") < order.index("configure_optimizers")
    assert order.index("before_train") < order.index("teardown")
    assert "after_train" in order
    assert "teardown" in order


def test_trainer_hooks_fire_after_each_step():
    step_losses = []
    hooks = HookManager()
    hooks.register("after_step", lambda ctx: step_losses.append(ctx.loss))
    strategy = _make_strategy({"loss": torch.tensor(0.5, requires_grad=True)})

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        with patch("os.makedirs"):
            MockDL.return_value = [{"x": 1}, {"x": 2}, {"x": 3}]
            trainer = Trainer(
                model=MagicMock(),
                dataset=MagicMock(),
                strategy=strategy,
                config=_make_config(),
                hooks=hooks,
            )
            trainer.run()

    assert len(step_losses) == 3
    assert all(l == 0.5 for l in step_losses)


def test_trainer_saves_model_to_output_dir():
    model = MagicMock()
    strategy = _make_strategy()

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        with patch("os.makedirs"):
            MockDL.return_value = []
            trainer = Trainer(
                model=model,
                dataset=MagicMock(),
                strategy=strategy,
                config=_make_config(),
            )
            trainer.run()

    model.save.assert_called_once_with("/tmp/test_out")
