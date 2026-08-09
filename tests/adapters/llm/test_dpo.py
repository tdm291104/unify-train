import pytest
import torch
from unittest.mock import MagicMock, patch
from core.base.io_types import TEXT_TO_TEXT, IOType, DataType


def _make_logits(batch_size: int = 1, seq_len: int = 4, vocab_size: int = 100) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)


def _make_batch(batch_size: int = 1, seq_len: int = 4) -> dict:
    ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = ids.clone()
    seq = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    return {"chosen": seq, "rejected": seq}


def _make_model(batch_size: int = 1, seq_len: int = 4, vocab_size: int = 100):
    model = MagicMock()
    model.raw_model = MagicMock()
    model.raw_model.parameters.return_value = iter([torch.randn(4, 4, requires_grad=True)])
    logits = _make_logits(batch_size, seq_len, vocab_size)
    model.forward.return_value = {"logits": logits}

    # raw_model(input_ids=..., attention_mask=...) -> output with .logits
    raw_out = MagicMock()
    raw_out.logits = _make_logits(batch_size, seq_len, vocab_size).detach()
    model.raw_model.return_value = raw_out
    return model


def test_dpo_registers():
    import adapters.llm.strategies.dpo
    from core.registry import STRATEGIES
    assert "dpo" in STRATEGIES.list_all()


def test_dpo_compatible_io_types():
    from adapters.llm.strategies.dpo import DPOStrategy
    assert TEXT_TO_TEXT in DPOStrategy.compatible_io_types


def test_dpo_setup_creates_ref_model():
    from adapters.llm.strategies.dpo import DPOStrategy
    model = _make_model()
    peft_mock = MagicMock()
    with patch("adapters.llm.strategies.dpo.get_peft_model", return_value=peft_mock), \
         patch("adapters.llm.strategies.dpo.LoraConfig"):
        strategy = DPOStrategy()
        strategy.setup(model, {})
    assert hasattr(strategy, "_ref_model")


def test_dpo_setup_ref_model_frozen():
    from adapters.llm.strategies.dpo import DPOStrategy
    model = _make_model()
    peft_mock = MagicMock()
    # Use a real small tensor to check requires_grad
    real_param = torch.nn.Parameter(torch.randn(3))
    ref_mock = MagicMock()
    ref_mock.parameters.return_value = iter([real_param])

    with patch("adapters.llm.strategies.dpo.get_peft_model", return_value=peft_mock), \
         patch("adapters.llm.strategies.dpo.LoraConfig"), \
         patch("adapters.llm.strategies.dpo.copy.deepcopy", return_value=ref_mock):
        strategy = DPOStrategy()
        strategy.setup(model, {})

    assert not real_param.requires_grad


def test_configure_optimizers_returns_adamw():
    from adapters.llm.strategies.dpo import DPOStrategy
    from torch.optim import AdamW
    model = _make_model()
    opt = DPOStrategy().configure_optimizers(model, {"lr": 1e-4})
    assert isinstance(opt, AdamW)


def test_training_step_returns_loss_tensor():
    from adapters.llm.strategies.dpo import DPOStrategy
    strategy = DPOStrategy()
    strategy._ref_model = MagicMock()
    strategy._beta = 0.1

    ref_out = MagicMock()
    ref_out.logits = _make_logits(1, 4, 100).detach()
    strategy._ref_model.return_value = ref_out

    model = _make_model(batch_size=1, seq_len=4, vocab_size=100)
    result = strategy.training_step(model, _make_batch(1, 4), step=0)
    assert "loss" in result
    assert isinstance(result["loss"], torch.Tensor)


def test_teardown_merges_and_cleans_ref():
    from adapters.llm.strategies.dpo import DPOStrategy
    strategy = DPOStrategy()
    strategy._ref_model = MagicMock()

    model = _make_model()
    merged = MagicMock()
    model.raw_model.merge_and_unload.return_value = merged
    strategy.teardown(model)

    model.raw_model.merge_and_unload.assert_called_once()
    assert model._model is merged
    assert not hasattr(strategy, "_ref_model")


def test_validate_rejects_image_io():
    from adapters.llm.strategies.dpo import DPOStrategy
    with pytest.raises(ValueError, match="does not support io_type"):
        DPOStrategy().validate(IOType(DataType.IMAGE, DataType.CATEGORY), None)


def test_dpo_loss_formula():
    """Verify the DPO loss formula using known inputs."""
    from adapters.llm.strategies.dpo import DPOStrategy
    import torch.nn.functional as F

    strategy = DPOStrategy()
    strategy._beta = 1.0

    # Craft logits so we can compute expected loss analytically
    B, T, V = 1, 3, 4
    vocab_size = V
    chosen_logits = torch.zeros(B, T, V, requires_grad=True)
    rejected_logits = torch.zeros(B, T, V, requires_grad=True)

    labels = torch.ones(B, T, dtype=torch.long)

    # With uniform logits, log_probs = log(1/V) for all tokens
    # So policy log-ratio = 0, ref log-ratio = 0, loss = -logsigmoid(0) = log(2)
    ref_model = MagicMock()
    ref_out = MagicMock()
    ref_out.logits = chosen_logits.clone()
    ref_model.return_value = ref_out

    strategy._ref_model = ref_model

    model = MagicMock()
    # First call returns chosen logits, second returns rejected logits
    model.forward.side_effect = [
        {"logits": chosen_logits.clone()},
        {"logits": rejected_logits.clone()},
    ]

    batch = {
        "chosen": {"input_ids": labels.clone(), "attention_mask": torch.ones(B, T, dtype=torch.long), "labels": labels.clone()},
        "rejected": {"input_ids": labels.clone(), "attention_mask": torch.ones(B, T, dtype=torch.long), "labels": labels.clone()},
    }

    result = strategy.training_step(model, batch, step=0)

    import math
    expected = math.log(2)
    assert abs(result["loss"].item() - expected) < 1e-4
