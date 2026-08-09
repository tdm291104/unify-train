# unify-train

A lightweight ML training framework for training and fine-tuning models via a single YAML config.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
unify-train train --config configs/my_run.yaml
unify-train train --config configs/my_run.yaml --resume outputs/checkpoint.pt
```

### Example config

```yaml
model:
  name: distilgpt2
  params:
    pretrained: distilgpt2

dataset:
  name: text_clm
  params:
    pretrained: distilgpt2
    max_length: 128
    texts:
      - "The quick brown fox jumps over the lazy dog."

strategy:
  name: lora
  params:
    r: 8
    lora_alpha: 16
    lr: 3e-4
    scheduler: cosine

evaluator:
  name: perplexity

io_type:
  input: text
  output: text

train:
  max_epochs: 3
  batch_size: 4
  device: cuda
  fp16: true
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0

output_dir: outputs/my_run
```

## Built-in adapters

| Type | Available |
|---|---|
| Models | `distilgpt2`, `resnet18` |
| Strategies | `lora`, `sft`, `ia3`, `dpo`, `supervised` |
| Datasets | `text_clm`, `hf_text`, `hf_dpo`, `synthetic_image` |
| Evaluators | `perplexity`, `accuracy` |

## Extending

Register custom components via decorator:

```python
from core.base.model import BaseModel
from core.base.io_types import TEXT_TO_TEXT
from core.registry import MODELS

@MODELS.register("my_model")
class MyModel(BaseModel):
    io_type = TEXT_TO_TEXT
    ...
```

Same pattern for `STRATEGIES`, `DATASETS`, `EVALUATORS`.

## Development

```bash
pytest
```
