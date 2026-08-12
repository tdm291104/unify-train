import pytest
import torch
from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from core.cli.main import main


def _write_config(tmp_path, model_name="hf_causal_lm", output_dir=None):
    if output_dir is None:
        out = tmp_path / "outputs"
        out.mkdir(exist_ok=True)
        output_dir = str(out)
    cfg_text = f"""
model:
  name: {model_name}
  params:
    pretrained: gpt2
dataset:
  name: text_clm
  params:
    pretrained: gpt2
    max_length: 32
    texts: ["hello"]
strategy:
  name: lora
  params:
    r: 4
    lr: 3e-4
io_type:
  input: text
  output: text
output_dir: {output_dir}
"""
    p = tmp_path / "config.yaml"
    p.write_text(cfg_text)
    return str(p)


def _make_mock_model():
    """Return a fully wired mock model ready for happy-path tests."""
    mock_tok = MagicMock()
    mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_tok.eos_token_id = 0
    mock_tok.decode.return_value = "hello world"

    mock_model = MagicMock()
    mock_model.tokenizer = mock_tok
    mock_model.raw_model = MagicMock()
    # generate returns prompt tokens + 2 new tokens
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 42, 43]])

    mock_cls = MagicMock(return_value=mock_model)
    return mock_cls, mock_model, mock_tok


def test_infer_generates_output(tmp_path):
    runner = CliRunner()
    cfg_path = _write_config(tmp_path)
    mock_cls, mock_model, mock_tok = _make_mock_model()

    with patch("core.registry.MODELS") as mock_reg:
        mock_reg.get.return_value = mock_cls
        result = runner.invoke(main, ["infer", cfg_path, "--prompt", "say hi"])

    assert result.exit_code == 0, result.output
    assert "hello world" in result.output


def test_infer_missing_tokenizer_raises(tmp_path):
    runner = CliRunner()
    cfg_path = _write_config(tmp_path)

    mock_model = MagicMock()
    mock_model.tokenizer = None  # triggers the guard
    mock_cls = MagicMock(return_value=mock_model)

    with patch("core.registry.MODELS") as mock_reg:
        mock_reg.get.return_value = mock_cls
        result = runner.invoke(main, ["infer", cfg_path, "--prompt", "say hi"])

    assert result.exit_code != 0
    assert "does not expose a tokenizer" in result.output


def test_infer_uses_checkpoint_dir(tmp_path):
    runner = CliRunner()
    cfg_path = _write_config(tmp_path, output_dir=str(tmp_path / "outputs"))
    mock_cls, mock_model, _ = _make_mock_model()

    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()

    with patch("core.registry.MODELS") as mock_reg:
        mock_reg.get.return_value = mock_cls
        result = runner.invoke(
            main,
            ["infer", cfg_path, "--prompt", "say hi", "--checkpoint", str(ckpt_dir)],
        )

    assert result.exit_code == 0, result.output
    mock_model.load.assert_called_once_with(str(ckpt_dir))


def test_infer_uses_output_dir_when_no_checkpoint(tmp_path):
    runner = CliRunner()
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(exist_ok=True)
    cfg_path = _write_config(tmp_path, output_dir=str(out_dir))
    mock_cls, mock_model, _ = _make_mock_model()

    with patch("core.registry.MODELS") as mock_reg:
        mock_reg.get.return_value = mock_cls
        result = runner.invoke(main, ["infer", cfg_path, "--prompt", "say hi"])

    assert result.exit_code == 0, result.output
    mock_model.load.assert_called_once_with(str(out_dir))


def test_infer_skips_prompt_tokens_in_decode(tmp_path):
    runner = CliRunner()
    cfg_path = _write_config(tmp_path)

    mock_tok = MagicMock()
    # 3 prompt tokens
    mock_tok.return_value = {"input_ids": torch.tensor([[10, 20, 30]])}
    mock_tok.eos_token_id = 0
    mock_tok.decode.return_value = "generated text"

    mock_model = MagicMock()
    mock_model.tokenizer = mock_tok
    mock_model.raw_model = MagicMock()
    # 3 prompt + 2 generated tokens
    mock_model.generate.return_value = torch.tensor([[10, 20, 30, 99, 88]])

    mock_cls = MagicMock(return_value=mock_model)

    with patch("core.registry.MODELS") as mock_reg:
        mock_reg.get.return_value = mock_cls
        result = runner.invoke(main, ["infer", cfg_path, "--prompt", "say hi"])

    assert result.exit_code == 0, result.output

    # decode must have been called with only the 2 new tokens, not the full sequence
    decode_call_args = mock_tok.decode.call_args
    decoded_tensor = decode_call_args.args[0]
    assert torch.equal(decoded_tensor, torch.tensor([99, 88])), (
        f"Expected decode called with [99, 88], got {decoded_tensor}"
    )


def test_infer_invalid_temperature(tmp_path):
    runner = CliRunner()
    cfg_path = _write_config(tmp_path)
    result = runner.invoke(main, ["infer", cfg_path, "--prompt", "hi", "--temperature", "0.0"])
    assert result.exit_code != 0
    assert "temperature" in result.output.lower()


def test_infer_missing_output_dir(tmp_path):
    runner = CliRunner()
    # Point output_dir to a non-existent directory without creating it
    cfg_path = _write_config(tmp_path, output_dir=str(tmp_path / "nonexistent"))
    result = runner.invoke(main, ["infer", cfg_path, "--prompt", "hi"])
    assert result.exit_code != 0
    assert "does not exist" in result.output
