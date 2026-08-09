import pytest
import torch
from unittest.mock import MagicMock, patch, call
from core.config.schema import UnifyTrainConfig
from core.trainer.trainer import Trainer


def _make_config(**overrides):
    base = dict(
        model={"name": "m"},
        dataset={"name": "d"},
        strategy={"name": "s"},
        io_type={"input": "text", "output": "text"},
        output_dir="/tmp/test_amp",
    )
    base.update(overrides)
    return UnifyTrainConfig(**base)


def _make_trainer(config, batches=None, loss_val=0.5):
    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.side_effect = lambda m, c: m
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None
    strategy.training_step.return_value = {"loss": torch.tensor(loss_val, requires_grad=True)}

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(3, requires_grad=True)])

    batches = batches or [[{}]]

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = batches
        trainer = Trainer(
            model=model,
            dataset=MagicMock(),
            strategy=strategy,
            config=config,
        )
    return trainer, strategy, model


# --- Gradient accumulation ---

def test_optimizer_step_every_accum_steps():
    config = _make_config(
        train={"max_epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 2},
    )
    batches = [{}, {}, {}, {}]  # 4 batches, accum=2 → 2 optimizer steps
    trainer, strategy, model = _make_trainer(config, batches)
    optimizer = strategy.configure_optimizers.return_value

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = batches
        trainer.run()

    assert optimizer.step.call_count == 2


def test_gradient_accumulation_fires_after_step_correctly():
    config = _make_config(
        train={"max_epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 3},
    )
    batches = [{}, {}, {}, {}, {}, {}]  # 6 batches, accum=3 → 2 optimizer steps
    step_counts = []
    from core.hook.hooks import HookManager
    hooks = HookManager()
    hooks.register("after_step", lambda ctx: step_counts.append(ctx.step))

    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.side_effect = lambda m, c: m
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None
    strategy.training_step.return_value = {"loss": torch.tensor(0.3, requires_grad=True)}

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(3)])

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = batches
        trainer = Trainer(model=model, dataset=MagicMock(), strategy=strategy, config=config, hooks=hooks)
        trainer.run()

    assert len(step_counts) == 2  # 6 batches / 3 accum = 2 optimizer steps


def test_pending_step_at_epoch_end():
    """Leftover batches after last accum boundary are flushed at epoch end."""
    config = _make_config(
        train={"max_epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 4},
    )
    batches = [{}, {}, {}]  # 3 batches, accum=4 → 0 mid-epoch steps, 1 flush at end
    step_counts = []
    from core.hook.hooks import HookManager
    hooks = HookManager()
    hooks.register("after_step", lambda ctx: step_counts.append(ctx.step))

    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.side_effect = lambda m, c: m
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None
    strategy.training_step.return_value = {"loss": torch.tensor(0.2, requires_grad=True)}

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(3)])

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = batches
        trainer = Trainer(model=model, dataset=MagicMock(), strategy=strategy, config=config, hooks=hooks)
        trainer.run()

    assert len(step_counts) == 1


def test_accum_1_fires_every_batch():
    config = _make_config(
        train={"max_epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 1},
    )
    batches = [{}, {}, {}]
    step_counts = []
    from core.hook.hooks import HookManager
    hooks = HookManager()
    hooks.register("after_step", lambda ctx: step_counts.append(ctx.step))

    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.side_effect = lambda m, c: m
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None
    strategy.training_step.return_value = {"loss": torch.tensor(0.1, requires_grad=True)}

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(3)])

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = batches
        trainer = Trainer(model=model, dataset=MagicMock(), strategy=strategy, config=config, hooks=hooks)
        trainer.run()

    assert len(step_counts) == 3


def test_loss_scaled_by_accum_steps():
    """backward is called on loss/accum_steps, not the raw loss."""
    config = _make_config(
        train={"max_epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 4},
    )
    backward_inputs = []

    raw_loss = torch.tensor(0.8, requires_grad=True)

    class _CaptureLoss(torch.Tensor):
        pass

    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.side_effect = lambda m, c: m
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None
    strategy.training_step.return_value = {"loss": raw_loss}

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(3)])

    original_backward = torch.Tensor.backward

    def capture_backward(self, *args, **kwargs):
        backward_inputs.append(self.item())
        return original_backward(self, *args, **kwargs)

    with patch("core.trainer.trainer.DataLoader") as MockDL, \
         patch.object(torch.Tensor, "backward", capture_backward):
        MockDL.return_value = [{}]
        trainer = Trainer(model=model, dataset=MagicMock(), strategy=strategy, config=config)
        trainer.run()

    # The value passed to backward should be loss/accum_steps = 0.8/4 = 0.2
    assert len(backward_inputs) == 1
    assert abs(backward_inputs[0] - 0.2) < 1e-5


# --- Gradient clipping ---

def test_grad_norm_clipping_called_when_set():
    config = _make_config(
        train={"max_epochs": 1, "batch_size": 1, "max_grad_norm": 1.0},
    )
    with patch("core.trainer.trainer.DataLoader") as MockDL, \
         patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
        MockDL.return_value = [{}]
        trainer, strategy, model = _make_trainer(config, batches=[{}])
        trainer.run()

    mock_clip.assert_called()
    args = mock_clip.call_args[0]
    assert args[1] == 1.0


def test_grad_norm_not_called_when_zero():
    config = _make_config(
        train={"max_epochs": 1, "batch_size": 1, "max_grad_norm": 0.0},
    )
    with patch("core.trainer.trainer.DataLoader") as MockDL, \
         patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
        MockDL.return_value = [{}]
        trainer, strategy, model = _make_trainer(config, batches=[{}])
        trainer.run()

    mock_clip.assert_not_called()


# --- Config defaults ---

def test_train_config_fp16_default():
    from core.config.schema import TrainConfig
    cfg = TrainConfig()
    assert cfg.fp16 is False
    assert cfg.bf16 is False
    assert cfg.max_grad_norm == 0.0
    assert cfg.gradient_accumulation_steps == 1
