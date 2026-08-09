import math
import pytest
from core.base.io_types import TEXT_TO_TEXT


def test_perplexity_registers():
    import adapters.llm.evaluators.perplexity
    from core.registry import EVALUATORS
    assert "perplexity" in EVALUATORS.list_all()


def test_perplexity_io_type():
    import adapters.llm.evaluators.perplexity
    from core.registry import EVALUATORS
    assert EVALUATORS.get("perplexity").io_type == TEXT_TO_TEXT


def test_perplexity_compute_empty_returns_inf():
    from adapters.llm.evaluators.perplexity import PerplexityEvaluator
    result = PerplexityEvaluator().compute([], [])
    assert result["perplexity"] == float("inf")


def test_perplexity_compute_zero_loss():
    from adapters.llm.evaluators.perplexity import PerplexityEvaluator
    result = PerplexityEvaluator().compute([0.0, 0.0], [])
    assert abs(result["perplexity"] - 1.0) < 1e-6


def test_perplexity_compute_known_value():
    from adapters.llm.evaluators.perplexity import PerplexityEvaluator
    losses = [2.0, 4.0]  # avg = 3.0 → ppl = e^3
    result = PerplexityEvaluator().compute(losses, [])
    assert abs(result["perplexity"] - math.exp(3.0)) < 1e-5


def test_llm_init_registers_all():
    import adapters.llm
    from core.registry import MODELS, DATASETS, STRATEGIES, EVALUATORS
    assert "distilgpt2" in MODELS.list_all()
    assert "text_clm" in DATASETS.list_all()
    assert "lora" in STRATEGIES.list_all()
    assert "perplexity" in EVALUATORS.list_all()
