import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from core.cli.main import main


def test_help_exits_zero():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "unify-train" in result.output


def test_train_help():
    result = CliRunner().invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "--checkpoint-every" in result.output
    assert "--early-stopping-patience" in result.output


def test_list_adapters_help():
    result = CliRunner().invoke(main, ["list-adapters", "--help"])
    assert result.exit_code == 0


def test_list_adapters_shows_all_sections():
    result = CliRunner().invoke(main, ["list-adapters"])
    assert result.exit_code == 0
    assert "Models" in result.output
    assert "Datasets" in result.output
    assert "Strategies" in result.output
    assert "Evaluators" in result.output


def test_list_adapters_shows_registered_names():
    result = CliRunner().invoke(main, ["list-adapters"])
    assert result.exit_code == 0
    assert "distilgpt2" in result.output
    assert "resnet18" in result.output
    assert "lora" in result.output
    assert "supervised" in result.output
    assert "hf_text" in result.output
    assert "text_clm" in result.output
    assert "perplexity" in result.output
    assert "accuracy" in result.output


def test_train_missing_config_exits_nonzero():
    result = CliRunner().invoke(main, ["train", "nonexistent.yaml"])
    assert result.exit_code != 0


def _write_config(tmp_path, outdir=None):
    outdir = outdir or str(tmp_path / "out")
    cfg = tmp_path / "test.yaml"
    cfg.write_text(f"""
model:
  name: distilgpt2
dataset:
  name: text_clm
  params:
    texts: ["hello world"]
strategy:
  name: lora
io_type:
  input: text
  output: text
train:
  max_epochs: 1
  batch_size: 1
output_dir: {outdir}
""")
    return cfg


def test_train_runs_with_mock(tmp_path):
    cfg_file = _write_config(tmp_path)
    mock_trainer = MagicMock()
    with patch("core.trainer.trainer.build_trainer_from_config", return_value=mock_trainer):
        result = CliRunner().invoke(main, ["train", str(cfg_file)])
    assert mock_trainer.run.called


def test_train_no_checkpoint_flag(tmp_path):
    cfg_file = _write_config(tmp_path)
    mock_trainer = MagicMock()
    with patch("core.trainer.trainer.build_trainer_from_config", return_value=mock_trainer):
        result = CliRunner().invoke(main, ["train", str(cfg_file), "--no-checkpoint"])
    assert mock_trainer.run.called


def test_train_resume_flag(tmp_path):
    cfg_file = _write_config(tmp_path)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    mock_trainer = MagicMock()
    with patch("core.trainer.trainer.build_trainer_from_config", return_value=mock_trainer) as mock_build:
        result = CliRunner().invoke(main, ["train", str(cfg_file), "--resume", str(ckpt_dir)])
    assert mock_trainer.run.called
    call_kwargs = mock_build.call_args.kwargs
    assert call_kwargs.get("resume") == str(ckpt_dir)


def test_train_resume_nonexistent_dir_fails(tmp_path):
    cfg_file = _write_config(tmp_path)
    result = CliRunner().invoke(main, ["train", str(cfg_file), "--resume", "/nonexistent/ckpt"])
    assert result.exit_code != 0
