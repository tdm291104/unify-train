# unify-train

A lightweight ML training framework for training and fine-tuning models via a single YAML config.

## Install

```bash
# Core
pip install -e ".[dev]"

# Optional — install only what you need
pip install -e ".[qlora]"             # QLoRA (4-bit quantization)
pip install -e ".[generation-eval]"   # BLEU + ROUGE-L evaluator
pip install -e ".[classification-eval]" # F1/precision/recall evaluator
pip install -e ".[all-extras]"        # All of the above
```

## Usage

```bash
unify-train train --config configs/my_run.yaml
unify-train train --config configs/my_run.yaml --resume outputs/checkpoint.pt
```

### Example config — LLaMA + QLoRA + instruction data

```yaml
model:
  name: hf_causal_lm
  params:
    pretrained: meta-llama/Llama-3.2-1B
    load_in_4bit: true

dataset:
  name: instruction
  params:
    path: tatsu-lab/alpaca
    pretrained: meta-llama/Llama-3.2-1B
    max_length: 512

strategy:
  name: qlora
  params:
    r: 16
    lora_alpha: 32
    lr: 2e-4

evaluator:
  name: generation
  params:
    metrics: [bleu, rouge_l]

io_type:
  input: text
  output: text

train:
  max_epochs: 3
  batch_size: 4
  device: cuda
  bf16: true
  gradient_accumulation_steps: 8
  max_grad_norm: 1.0

output_dir: outputs/llama-qlora
```

### Example config — LoRA fine-tuning (minimal)

```yaml
model:
  name: hf_causal_lm
  params:
    pretrained: gpt2

dataset:
  name: text_clm
  params:
    pretrained: gpt2
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

output_dir: outputs/gpt2-lora
```

## Built-in adapters

| Type | Available |
|---|---|
| Models | `hf_causal_lm`, `distilgpt2`, `resnet18` |
| Strategies | `lora`, `qlora`, `sft`, `ia3`, `dpo`, `orpo`, `supervised` |
| Datasets | `instruction`, `text_clm`, `hf_text`, `hf_dpo`, `synthetic_image` |
| Evaluators | `generation`, `classification`, `perplexity`, `accuracy` |

### Adapter notes

| Adapter | Notes |
|---|---|
| `hf_causal_lm` | Any HuggingFace causal LM (LLaMA, Mistral, Phi, GPT-2, …) |
| `qlora` | Requires `.[qlora]` + CUDA. Extends `lora` — supports all LoRA params |
| `orpo` | ORPO alignment (no reference model). Uses `hf_dpo` dataset format |
| `instruction` | Alpaca and ShareGPT formats. Auto-detects format if not specified |
| `generation` | BLEU + ROUGE-L. Requires `.[generation-eval]` |
| `classification` | F1, precision, recall, exact-match. Requires `.[classification-eval]` |

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
uv run pytest
```
