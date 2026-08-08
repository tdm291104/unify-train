import pytest
import yaml
from core.config.schema import UnifyTrainConfig
from core.config.loader import load_config

VALID_DICT = {
    "model": {"name": "distilgpt2", "params": {"pretrained": "distilgpt2"}},
    "dataset": {"name": "text_clm", "params": {"pretrained": "distilgpt2", "texts": ["hello"]}},
    "strategy": {"name": "lora", "params": {"r": 8}},
    "evaluator": {"name": "perplexity"},
    "io_type": {"input": "text", "output": "text"},
    "train": {"max_epochs": 2, "batch_size": 4},
    "output_dir": "outputs/test",
}


def test_valid_config_parses():
    cfg = UnifyTrainConfig(**VALID_DICT)
    assert cfg.model.name == "distilgpt2"
    assert cfg.strategy.name == "lora"
    assert cfg.io_type.input == "text"
    assert cfg.train.max_epochs == 2


def test_invalid_io_type_input_raises():
    bad = {**VALID_DICT, "io_type": {"input": "not_a_type", "output": "text"}}
    with pytest.raises(Exception):
        UnifyTrainConfig(**bad)


def test_invalid_io_type_output_raises():
    bad = {**VALID_DICT, "io_type": {"input": "text", "output": "not_a_type"}}
    with pytest.raises(Exception):
        UnifyTrainConfig(**bad)


def test_defaults_applied():
    minimal = {
        "model": {"name": "distilgpt2"},
        "dataset": {"name": "text_clm"},
        "strategy": {"name": "lora"},
        "evaluator": {"name": "perplexity"},
        "io_type": {"input": "text", "output": "text"},
    }
    cfg = UnifyTrainConfig(**minimal)
    assert cfg.train.max_epochs == 3
    assert cfg.train.batch_size == 4
    assert cfg.task is None
    assert cfg.output_dir == "outputs/"
    assert cfg.model.params == {}
    assert cfg.strategy.params == {}


def test_task_field_accepts_string():
    cfg = UnifyTrainConfig(**{**VALID_DICT, "task": "translation"})
    assert cfg.task == "translation"


def test_load_config_from_yaml(tmp_path):
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(yaml.dump(VALID_DICT))
    cfg = load_config(str(yaml_file))
    assert cfg.model.name == "distilgpt2"
    assert cfg.io_type.input == "text"


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nonexistent.yaml"))
