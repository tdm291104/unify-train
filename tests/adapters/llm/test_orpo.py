import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock


def _make_logits(B=2, T=5, V=8, fill=0.0):
    return torch.full((B, T, V), fill)


def _make_labels(B=2, T=5, token=1):
    labels = torch.full((B, T), -100, dtype=torch.long)
    labels[:, 1:] = token
    return labels


def _make_batch(B=2, T=5, V=8):
    ids = torch.ones(B, T, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.long)
    labels = _make_labels(B, T)
    seq = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    return {"chosen": seq, "rejected": seq}


def test_registers():
    import adapters.llm.strategies.orpo  # noqa: F401
    from core.registry import STRATEGIES
    assert "orpo" in STRATEGIES.list_all()


def test_setup_sets_lambda():
    from adapters.llm.strategies.orpo import ORPOStrategy
    strat = ORPOStrategy()
    mock_model = MagicMock()
    mock_model.raw_model = MagicMock()
    strat.setup(mock_model, {"lambda_orpo": 0.5})
    assert strat._lambda_orpo == 0.5


def test_setup_default_lambda():
    from adapters.llm.strategies.orpo import ORPOStrategy
    strat = ORPOStrategy()
    mock_model = MagicMock()
    mock_model.raw_model = MagicMock()
    strat.setup(mock_model, {})
    assert strat._lambda_orpo == 0.1


def test_training_step_returns_loss_tensor():
    from adapters.llm.strategies.orpo import ORPOStrategy

    B, T, V = 2, 5, 8
    logits = _make_logits(B, T, V)
    labels = _make_labels(B, T)
    chosen_loss = torch.tensor(1.5, requires_grad=True)

    mock_model = MagicMock()
    mock_model.forward.side_effect = [
        {"loss": chosen_loss, "logits": logits},
        {"loss": torch.tensor(2.0), "logits": logits},
    ]

    batch = _make_batch(B, T, V)
    strat = ORPOStrategy()
    strat._lambda_orpo = 1.0

    result = strat.training_step(mock_model, batch, step=0)

    assert "loss" in result
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].requires_grad
    assert "sft_loss" in result
    assert "or_loss" in result


def test_training_step_calls_model_twice():
    from adapters.llm.strategies.orpo import ORPOStrategy

    B, T, V = 2, 5, 8
    logits = _make_logits(B, T, V)

    mock_model = MagicMock()
    mock_model.forward.side_effect = [
        {"loss": torch.tensor(1.0, requires_grad=True), "logits": logits},
        {"loss": torch.tensor(1.5), "logits": logits},
    ]

    strat = ORPOStrategy()
    strat._lambda_orpo = 0.1
    strat.training_step(mock_model, _make_batch(B, T, V), step=0)

    assert mock_model.forward.call_count == 2


def test_configure_optimizers_returns_adamw():
    from adapters.llm.strategies.orpo import ORPOStrategy
    from torch.optim import AdamW

    mock_model = MagicMock()
    mock_model.raw_model.parameters.return_value = iter([torch.randn(4, 4, requires_grad=True)])

    opt = ORPOStrategy().configure_optimizers(mock_model, {"lr": 8e-6})
    assert isinstance(opt, AdamW)


def test_teardown_is_noop():
    from adapters.llm.strategies.orpo import ORPOStrategy
    strat = ORPOStrategy()
    mock_model = MagicMock()
    strat.teardown(mock_model)  # must not raise
    mock_model.assert_not_called()
