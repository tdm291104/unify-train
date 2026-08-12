import pytest
import torch
from unittest.mock import MagicMock, patch, call
from core.config.schema import UnifyTrainConfig
from core.trainer.trainer import Trainer


def _make_strategy(loss_val=0.5):
    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.return_value = MagicMock()
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None
    strategy.training_step.return_value = {"loss": torch.tensor(loss_val, requires_grad=True)}
    return strategy


def _make_config(**overrides):
    defaults = dict(
        model={"name": "m"},
        dataset={"name": "d"},
        strategy={"name": "s"},
        io_type={"input": "text", "output": "text"},
        train={"max_epochs": 2, "batch_size": 1},
        output_dir="/tmp/test_eval",
    )
    defaults.update(overrides)
    return UnifyTrainConfig(**defaults)


def _make_trainer(evaluator=None, eval_dataset=None, config=None, strategy=None):
    config = config or _make_config()
    strategy = strategy or _make_strategy()
    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = [[{"input_ids": torch.ones(1, 4, dtype=torch.long)}]]
        trainer = Trainer(
            model=model,
            dataset=MagicMock(),
            strategy=strategy,
            config=config,
            evaluator=evaluator,
            eval_dataset=eval_dataset,
        )
    return trainer, model, strategy, MockDL


def test_no_eval_when_evaluator_none():
    trainer, model, strategy, _ = _make_trainer()
    with patch("core.trainer.trainer.DataLoader"):
        trainer.run()
    # raw_model.eval() should NOT be called
    model.raw_model.eval.assert_not_called()


def test_no_eval_when_eval_dataset_none():
    evaluator = MagicMock()
    trainer, model, strategy, _ = _make_trainer(evaluator=evaluator, eval_dataset=None)
    with patch("core.trainer.trainer.DataLoader"):
        trainer.run()
    model.raw_model.eval.assert_not_called()


def test_eval_called_after_each_epoch():
    evaluator = MagicMock()
    evaluator.extract.return_value = ([0.5], [])
    evaluator.compute.return_value = {"perplexity": 1.65}

    eval_ds = MagicMock()
    eval_ds.collate_fn = None

    config = _make_config(train={"max_epochs": 2, "batch_size": 1})

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()
    model.forward.return_value = {"loss": torch.tensor(0.5), "logits": torch.randn(1, 4, 10)}

    strategy = _make_strategy()
    strategy.setup.return_value = model

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = [[{}]]
        trainer = Trainer(
            model=model, dataset=MagicMock(), strategy=strategy,
            config=config, evaluator=evaluator, eval_dataset=eval_ds,
        )
        trainer.run()

    # eval called once per epoch = 2 times
    assert model.raw_model.eval.call_count == 2
    assert model.raw_model.train.call_count == 2
    assert evaluator.compute.call_count == 2


def test_eval_every_n_epochs():
    evaluator = MagicMock()
    evaluator.extract.return_value = ([0.5], [])
    evaluator.compute.return_value = {"perplexity": 1.65}

    eval_ds = MagicMock()

    config = _make_config(
        train={"max_epochs": 4, "batch_size": 1},
        eval={"eval_every_n_epochs": 2},
    )

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()
    model.forward.return_value = {"loss": torch.tensor(0.5), "logits": torch.randn(1, 4, 10)}

    strategy = _make_strategy()
    strategy.setup.return_value = model

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = [[{}]]
        trainer = Trainer(
            model=model, dataset=MagicMock(), strategy=strategy,
            config=config, evaluator=evaluator, eval_dataset=eval_ds,
        )
        trainer.run()

    # eval_every_n_epochs=2 over 4 epochs → 2 eval runs (epoch 2 and epoch 4)
    assert evaluator.compute.call_count == 2


def test_eval_metrics_stored_in_ctx():
    evaluator = MagicMock()
    evaluator.extract.return_value = ([0.4], [])
    evaluator.compute.return_value = {"perplexity": 1.49}

    eval_ds = MagicMock()
    after_eval_ctx = []

    from core.hook.hooks import HookManager
    hooks = HookManager()
    hooks.register("after_eval", lambda ctx: after_eval_ctx.append(dict(ctx.metrics)))

    config = _make_config(train={"max_epochs": 1, "batch_size": 1})

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()
    model.forward.return_value = {"loss": torch.tensor(0.4), "logits": torch.randn(1, 4, 10)}

    strategy = _make_strategy()
    strategy.setup.return_value = model

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = [[{}]]
        trainer = Trainer(
            model=model, dataset=MagicMock(), strategy=strategy,
            config=config, evaluator=evaluator, eval_dataset=eval_ds,
            hooks=hooks,
        )
        trainer.run()

    assert len(after_eval_ctx) == 1
    assert after_eval_ctx[0]["perplexity"] == pytest.approx(1.49)


def test_eval_runs_with_no_grad():
    """Ensure no gradients are computed during eval."""
    evaluator = MagicMock()
    evaluator.extract.return_value = ([], [])
    evaluator.compute.return_value = {}

    eval_ds = MagicMock()
    grad_states = []

    config = _make_config(train={"max_epochs": 1, "batch_size": 1})

    def capturing_forward(batch):
        grad_states.append(torch.is_grad_enabled())
        return {"loss": torch.tensor(0.5), "logits": torch.randn(1, 4, 10)}

    model = MagicMock()
    model.save = MagicMock()
    model.raw_model = MagicMock()
    model.forward.side_effect = capturing_forward

    strategy = _make_strategy()
    strategy.setup.return_value = model
    strategy.training_step.return_value = {"loss": torch.tensor(0.5, requires_grad=True)}

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = [[{}]]
        trainer = Trainer(
            model=model, dataset=MagicMock(), strategy=strategy,
            config=config, evaluator=evaluator, eval_dataset=eval_ds,
        )
        trainer.run()

    # At least one forward call should have been inside no_grad
    assert any(not g for g in grad_states)


def test_perplexity_evaluator_extract():
    from adapters.llm.evaluators.perplexity import PerplexityEvaluator
    ev = PerplexityEvaluator()
    out = {"loss": torch.tensor(0.5)}
    preds, refs = ev.extract(out, {})
    assert preds == [pytest.approx(0.5)]
    assert refs == []


def test_accuracy_evaluator_extract():
    from adapters.vision.evaluators.accuracy import AccuracyEvaluator
    ev = AccuracyEvaluator()
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])  # preds: [1, 0]
    labels = torch.tensor([1, 0])
    preds, refs = ev.extract({"logits": logits}, {"labels": labels})
    assert preds == [1, 0]
    assert refs == [1, 0]


def test_eval_config_defaults():
    from core.config.schema import EvalConfig
    cfg = EvalConfig()
    assert cfg.eval_every_n_epochs == 1


def test_build_trainer_from_config_builds_eval_dataset(tmp_path):
    from core.trainer.trainer import build_trainer_from_config
    from core.config.schema import UnifyTrainConfig

    config = UnifyTrainConfig(
        model={"name": "m"},
        dataset={"name": "d"},
        strategy={"name": "s"},
        eval_dataset={"name": "eval_d"},
        evaluator={"name": "ev"},
        io_type={"input": "text", "output": "text"},
        output_dir=str(tmp_path),
    )

    mock_model = MagicMock()
    mock_dataset = MagicMock()
    mock_eval_dataset = MagicMock()
    mock_strategy = MagicMock()
    mock_evaluator = MagicMock()

    with patch("core.registry.MODELS") as MockMODELS, \
         patch("core.registry.DATASETS") as MockDATASETS, \
         patch("core.registry.STRATEGIES") as MockSTRATEGIES, \
         patch("core.registry.EVALUATORS") as MockEVALUATORS:
        MockMODELS.get.return_value = MagicMock(return_value=mock_model)
        MockDATASETS.get.side_effect = lambda name: (
            MagicMock(return_value=mock_dataset) if name == "d"
            else MagicMock(return_value=mock_eval_dataset)
        )
        MockSTRATEGIES.get.return_value = MagicMock(return_value=mock_strategy)
        MockEVALUATORS.get.return_value = MagicMock(return_value=mock_evaluator)

        trainer = build_trainer_from_config(config)

    assert trainer._evaluator is mock_evaluator
    assert trainer._eval_dataset is mock_eval_dataset


def test_evaluator_receives_params(tmp_path):
    """Evaluator constructor must receive config.evaluator.params."""
    received = {}

    from core.base.io_types import TEXT_TO_TEXT

    class ParamCapturingEvaluator:
        io_type = TEXT_TO_TEXT

        def __init__(self, params):
            received["params"] = params

        def extract(self, out, batch):
            return [], []

        def compute(self, preds, refs):
            return {}

    from core.registry import EVALUATORS
    from core.config.schema import UnifyTrainConfig
    from core.trainer.trainer import build_trainer_from_config

    EVALUATORS._registry["_capture"] = ParamCapturingEvaluator

    cfg = UnifyTrainConfig(
        model={"name": "_m"},
        dataset={"name": "_d"},
        strategy={"name": "_s"},
        evaluator={"name": "_capture", "params": {"foo": "bar"}},
        io_type={"input": "text", "output": "text"},
    )

    mock_model = MagicMock()
    mock_dataset = MagicMock()
    mock_strategy = MagicMock()

    try:
        with patch("core.registry.MODELS") as MockMODELS, \
             patch("core.registry.DATASETS") as MockDATASETS, \
             patch("core.registry.STRATEGIES") as MockSTRATEGIES:
            MockMODELS.get.return_value = MagicMock(return_value=mock_model)
            MockDATASETS.get.return_value = MagicMock(return_value=mock_dataset)
            MockSTRATEGIES.get.return_value = MagicMock(return_value=mock_strategy)
            build_trainer_from_config(cfg)
        assert received.get("params") == {"foo": "bar"}
    finally:
        del EVALUATORS._registry["_capture"]
