import pytest
import torch
from unittest.mock import MagicMock, patch, call
from core.base.strategy import BaseStrategy


class _ConcreteStrategy(BaseStrategy):
    compatible_io_types = []

    def setup(self, model, config): return model
    def configure_optimizers(self, model, config): return MagicMock()
    def training_step(self, model, batch, optimizer, step): return {"loss": 0.0}
    def teardown(self, model): pass


def test_base_strategy_configure_scheduler_returns_none():
    strat = _ConcreteStrategy()
    opt = MagicMock()
    assert strat.configure_scheduler(opt, {}) is None


def test_trainer_calls_configure_scheduler():
    from core.trainer.trainer import Trainer
    from core.config.schema import UnifyTrainConfig
    from unittest.mock import MagicMock

    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.return_value = MagicMock()
    strategy.configure_optimizers.return_value = MagicMock()
    mock_scheduler = MagicMock()
    strategy.configure_scheduler.return_value = mock_scheduler
    strategy.training_step.return_value = {"loss": 0.5}

    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=2)
    dataset.collate_fn = MagicMock(return_value={})
    dataset.__iter__ = MagicMock(return_value=iter([{}, {}]))

    config = UnifyTrainConfig(
        model={"name": "m"},
        dataset={"name": "d"},
        strategy={"name": "s"},
        io_type={"input": "text", "output": "text"},
        train={"max_epochs": 2, "batch_size": 2},
        output_dir="/tmp/test_sched",
    )

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = [[{}], [{}]]
        model = MagicMock()
        model.save = MagicMock()
        trainer = Trainer(model=model, dataset=dataset, strategy=strategy, config=config)
        trainer.run()

    strategy.configure_scheduler.assert_called_once()
    assert mock_scheduler.step.call_count == 2  # once per epoch


def test_trainer_skips_scheduler_step_when_none():
    from core.trainer.trainer import Trainer
    from core.config.schema import UnifyTrainConfig

    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.return_value = MagicMock()
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None
    strategy.training_step.return_value = {"loss": 0.1}

    config = UnifyTrainConfig(
        model={"name": "m"},
        dataset={"name": "d"},
        strategy={"name": "s"},
        io_type={"input": "text", "output": "text"},
        train={"max_epochs": 1, "batch_size": 2},
        output_dir="/tmp/test_no_sched",
    )

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = [[{}]]
        model = MagicMock()
        model.save = MagicMock()
        trainer = Trainer(model=model, dataset=MagicMock(), strategy=strategy, config=config)
        trainer.run()  # should not raise


def test_lora_configure_scheduler_none_when_not_set():
    from adapters.llm.strategies.lora import LoRAStrategy
    opt = MagicMock()
    assert LoRAStrategy().configure_scheduler(opt, {}) is None


def test_lora_configure_scheduler_cosine():
    from adapters.llm.strategies.lora import LoRAStrategy
    from torch.optim.lr_scheduler import CosineAnnealingLR
    params = [torch.randn(3, 3, requires_grad=True)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = LoRAStrategy().configure_scheduler(opt, {"scheduler": "cosine", "t_max": 5})
    assert isinstance(sched, CosineAnnealingLR)


def test_lora_configure_scheduler_step():
    from adapters.llm.strategies.lora import LoRAStrategy
    from torch.optim.lr_scheduler import StepLR
    params = [torch.randn(3, 3, requires_grad=True)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = LoRAStrategy().configure_scheduler(opt, {"scheduler": "step", "step_size": 2, "gamma": 0.5})
    assert isinstance(sched, StepLR)


def test_lora_cosine_scheduler_steps_lr():
    from adapters.llm.strategies.lora import LoRAStrategy
    params = [torch.randn(3, 3, requires_grad=True)]
    opt = torch.optim.AdamW(params, lr=1.0)
    sched = LoRAStrategy().configure_scheduler(opt, {"scheduler": "cosine", "t_max": 2})
    lr_before = opt.param_groups[0]["lr"]
    sched.step()
    lr_after = opt.param_groups[0]["lr"]
    assert lr_after != lr_before
