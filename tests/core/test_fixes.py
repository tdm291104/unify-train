"""Tests covering the 7 review fixes."""
import pytest
import torch
from unittest.mock import MagicMock, patch
from core.config.schema import UnifyTrainConfig, TrainConfig, EvaluatorConfig
from core.trainer.trainer import Trainer


# --- Fix 1: configure_scheduler deduplication ---

def test_base_strategy_cosine_scheduler():
    from core.base.strategy import BaseStrategy

    class _S(BaseStrategy):
        compatible_io_types = []
        def setup(self, m, c): return m
        def configure_optimizers(self, m, c): return MagicMock()
        def training_step(self, m, b, s): return {"loss": torch.tensor(0.0)}
        def teardown(self, m): pass

    params = [torch.randn(3, requires_grad=True)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    sched = _S().configure_scheduler(opt, {"scheduler": "cosine", "t_max": 5})
    assert isinstance(sched, CosineAnnealingLR)


def test_base_strategy_step_scheduler():
    from core.base.strategy import BaseStrategy
    from torch.optim.lr_scheduler import StepLR

    class _S(BaseStrategy):
        compatible_io_types = []
        def setup(self, m, c): return m
        def configure_optimizers(self, m, c): return MagicMock()
        def training_step(self, m, b, s): return {"loss": torch.tensor(0.0)}
        def teardown(self, m): pass

    params = [torch.randn(3, requires_grad=True)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = _S().configure_scheduler(opt, {"scheduler": "step", "step_size": 2, "gamma": 0.5})
    assert isinstance(sched, StepLR)


def test_lora_uses_base_configure_scheduler():
    from adapters.llm.strategies.lora import LoRAStrategy
    from core.base.strategy import BaseStrategy
    assert LoRAStrategy.configure_scheduler is BaseStrategy.configure_scheduler


def test_sft_uses_base_configure_scheduler():
    from adapters.llm.strategies.sft import SFTStrategy
    from core.base.strategy import BaseStrategy
    assert SFTStrategy.configure_scheduler is BaseStrategy.configure_scheduler


def test_ia3_uses_base_configure_scheduler():
    from adapters.llm.strategies.ia3 import IA3Strategy
    from core.base.strategy import BaseStrategy
    assert IA3Strategy.configure_scheduler is BaseStrategy.configure_scheduler


def test_dpo_uses_base_configure_scheduler():
    from adapters.llm.strategies.dpo import DPOStrategy
    from core.base.strategy import BaseStrategy
    assert DPOStrategy.configure_scheduler is BaseStrategy.configure_scheduler


# --- Fix 2: Device management ---

def test_train_config_device_default():
    cfg = TrainConfig()
    assert cfg.device == "cpu"


def test_trainer_moves_model_to_device():
    config = UnifyTrainConfig(
        model={"name": "m"}, dataset={"name": "d"}, strategy={"name": "s"},
        io_type={"input": "text", "output": "text"},
        train={"max_epochs": 1, "batch_size": 1, "device": "cpu"},
        output_dir="/tmp/test_device",
    )
    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.side_effect = lambda m, c: m
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None
    strategy.training_step.return_value = {"loss": torch.tensor(0.1, requires_grad=True)}

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = []
        trainer = Trainer(model=model, dataset=MagicMock(), strategy=strategy, config=config)
        trainer.run()

    model.raw_model.to.assert_called_once_with(torch.device("cpu"))


def test_to_device_moves_tensor():
    t = torch.tensor([1.0, 2.0])
    result = Trainer._to_device(t, torch.device("cpu"))
    assert isinstance(result, torch.Tensor)


def test_to_device_recurses_dict():
    batch = {"a": torch.tensor([1.0]), "b": {"c": torch.tensor([2.0])}}
    result = Trainer._to_device(batch, torch.device("cpu"))
    assert isinstance(result["a"], torch.Tensor)
    assert isinstance(result["b"]["c"], torch.Tensor)


def test_to_device_ignores_non_tensor():
    batch = {"text": "hello", "ids": torch.tensor([1])}
    result = Trainer._to_device(batch, torch.device("cpu"))
    assert result["text"] == "hello"


# --- Fix 3: DPOStrategy.__init__ ---

def test_dpo_ref_model_initialized_to_none():
    from adapters.llm.strategies.dpo import DPOStrategy
    strategy = DPOStrategy()
    assert strategy._ref_model is None
    assert strategy._beta == pytest.approx(0.1)


def test_dpo_setup_sets_ref_model():
    from adapters.llm.strategies.dpo import DPOStrategy
    strategy = DPOStrategy()
    assert strategy._ref_model is None

    model = MagicMock()
    peft_mock = MagicMock()
    with patch("adapters.llm.strategies.dpo.get_peft_model", return_value=peft_mock), \
         patch("adapters.llm.strategies.dpo.LoraConfig"):
        strategy.setup(model, {"beta": 0.2})

    assert strategy._ref_model is not None
    assert strategy._beta == pytest.approx(0.2)


# --- Fix 4: EvaluatorConfig.params ---

def test_evaluator_config_has_params():
    cfg = EvaluatorConfig(name="perplexity")
    assert cfg.params == {}


def test_evaluator_config_params_passable():
    cfg = EvaluatorConfig(name="accuracy", params={"topk": 5})
    assert cfg.params["topk"] == 5


# --- Fix 5: raw_model in BaseModel ABC ---

def test_base_model_raw_model_is_abstract():
    from core.base.model import BaseModel
    assert "raw_model" in {
        name for name in dir(BaseModel)
        if getattr(getattr(BaseModel, name, None), "__isabstractmethod__", False)
    }


def test_model_without_raw_model_cannot_instantiate():
    from core.base.model import BaseModel
    from core.base.io_types import TEXT_TO_TEXT

    class IncompleteModel(BaseModel):
        io_type = TEXT_TO_TEXT
        def build(self, config): pass
        def forward(self, batch): return {}
        def save(self, path): pass
        def load(self, path): pass
        # raw_model NOT implemented

    with pytest.raises(TypeError):
        IncompleteModel()


# --- Fix 6: Eval DataLoader cached ---

def test_eval_loader_cached_across_epochs():
    config = UnifyTrainConfig(
        model={"name": "m"}, dataset={"name": "d"}, strategy={"name": "s"},
        io_type={"input": "text", "output": "text"},
        train={"max_epochs": 3, "batch_size": 1},
        output_dir="/tmp/test_cache",
    )
    evaluator = MagicMock()
    evaluator.extract.return_value = ([], [])
    evaluator.compute.return_value = {}

    eval_ds = MagicMock()

    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.side_effect = lambda m, c: m
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None
    strategy.training_step.return_value = {"loss": torch.tensor(0.1, requires_grad=True)}

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()

    dl_call_count = []

    original_dl = __import__("torch.utils.data", fromlist=["DataLoader"]).DataLoader

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.side_effect = lambda *a, **kw: (dl_call_count.append(1), [])[1]
        trainer = Trainer(
            model=model, dataset=MagicMock(), strategy=strategy,
            config=config, evaluator=evaluator, eval_dataset=eval_ds,
        )
        trainer.run()

    # DataLoader called once for train loader + once for eval loader (cached)
    # So total = 2, not 4 (would be 4 if eval loader created every epoch)
    assert dl_call_count.count(1) == 2
