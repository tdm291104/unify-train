import json
import os
import pytest
import torch
from unittest.mock import MagicMock, patch
from core.trainer.checkpoint_utils import load_checkpoint


def _write_meta(path: str, epoch: int, step: int) -> None:
    with open(os.path.join(path, "checkpoint_meta.json"), "w") as f:
        json.dump({"epoch": epoch, "step": step}, f)


def test_load_checkpoint_calls_model_load(tmp_path):
    model = MagicMock()
    _write_meta(str(tmp_path), epoch=2, step=16)
    load_checkpoint(str(tmp_path), model)
    model.load.assert_called_once_with(str(tmp_path))


def test_load_checkpoint_returns_epoch_and_step(tmp_path):
    model = MagicMock()
    _write_meta(str(tmp_path), epoch=3, step=42)
    start_epoch, start_step = load_checkpoint(str(tmp_path), model)
    assert start_epoch == 3
    assert start_step == 42


def test_load_checkpoint_no_meta_returns_zeros(tmp_path):
    model = MagicMock()
    start_epoch, start_step = load_checkpoint(str(tmp_path), model)
    assert start_epoch == 0
    assert start_step == 0


def test_load_checkpoint_missing_dir_raises():
    model = MagicMock()
    with pytest.raises(FileNotFoundError, match="not found"):
        load_checkpoint("/nonexistent/path", model)


def test_trainer_starts_from_start_epoch():
    from core.trainer.trainer import Trainer
    from core.config.schema import UnifyTrainConfig

    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.return_value = MagicMock()
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None

    epochs_seen = []
    def training_step(model, batch, step):
        epochs_seen.append(step)
        return {"loss": torch.tensor(0.1, requires_grad=True)}
    strategy.training_step.side_effect = training_step

    config = UnifyTrainConfig(
        model={"name": "m"},
        dataset={"name": "d"},
        strategy={"name": "s"},
        io_type={"input": "text", "output": "text"},
        train={"max_epochs": 3, "batch_size": 1},
        output_dir="/tmp/test_resume",
    )

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = [[{}]]
        model = MagicMock()
        model.save = MagicMock()
        trainer = Trainer(
            model=model, dataset=MagicMock(), strategy=strategy,
            config=config, start_epoch=2, start_step=20,
        )
        trainer.run()

    # Only 1 epoch should run (epoch 2, since range(2, 3))
    assert strategy.training_step.call_count == 1


def test_trainer_global_step_continues_from_start_step():
    from core.trainer.trainer import Trainer
    from core.config.schema import UnifyTrainConfig

    strategy = MagicMock()
    strategy.validate.return_value = None
    strategy.setup.return_value = MagicMock()
    strategy.configure_optimizers.return_value = MagicMock()
    strategy.configure_scheduler.return_value = None

    steps_seen = []
    def training_step(model, batch, step):
        steps_seen.append(step)
        return {"loss": torch.tensor(0.1, requires_grad=True)}
    strategy.training_step.side_effect = training_step

    config = UnifyTrainConfig(
        model={"name": "m"},
        dataset={"name": "d"},
        strategy={"name": "s"},
        io_type={"input": "text", "output": "text"},
        train={"max_epochs": 2, "batch_size": 1},
        output_dir="/tmp/test_step",
    )

    with patch("core.trainer.trainer.DataLoader") as MockDL:
        MockDL.return_value = [[{}]]
        model = MagicMock()
        model.save = MagicMock()
        trainer = Trainer(
            model=model, dataset=MagicMock(), strategy=strategy,
            config=config, start_epoch=1, start_step=8,
        )
        trainer.run()

    # First call to training_step should pass step=8
    assert steps_seen[0] == 8


def test_build_trainer_from_config_with_resume(tmp_path):
    from core.trainer.trainer import build_trainer_from_config
    from core.config.schema import UnifyTrainConfig

    _write_meta(str(tmp_path), epoch=1, step=5)

    config = UnifyTrainConfig(
        model={"name": "distilgpt2"},
        dataset={"name": "text_clm", "params": {"texts": ["hi"]}},
        strategy={"name": "lora"},
        io_type={"input": "text", "output": "text"},
        output_dir="/tmp/test_build_resume",
    )

    mock_model = MagicMock()

    with patch("core.registry.MODELS") as MockMODELS, \
         patch("core.registry.DATASETS") as MockDATASETS, \
         patch("core.registry.STRATEGIES") as MockSTRATEGIES, \
         patch("core.registry.EVALUATORS"):
        MockMODELS.get.return_value = MagicMock(return_value=mock_model)
        MockDATASETS.get.return_value = MagicMock(return_value=MagicMock())
        MockSTRATEGIES.get.return_value = MagicMock(return_value=MagicMock())

        trainer = build_trainer_from_config(config, resume=str(tmp_path))

    assert trainer._start_epoch == 1
    assert trainer._start_step == 5
    mock_model.load.assert_called_once_with(str(tmp_path))
