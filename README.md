# unify-train

[![Tests](https://github.com/tdm291104/unify-train/actions/workflows/test.yml/badge.svg)](https://github.com/tdm291104/unify-train/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A lightweight, config-driven ML training framework. Define your entire training run — model, dataset, strategy, evaluator — in a single YAML file and run it with one command.

---

## Why unify-train?

Most ML training code is tangled: model loading, optimizer setup, training loops, and evaluation logic are all mixed together and hard to swap out. When you want to switch from LoRA to QLoRA, or from perplexity to BLEU evaluation, you end up rewriting boilerplate.

`unify-train` separates *what* you train from *how* you train it. Each component — model, strategy, dataset, evaluator — is an independent adapter registered by name. You compose them in YAML, not code. Switching from SFT to ORPO is one line change.

---

## Features

- **YAML-first** — entire training run defined in one config file, no Python required
- **Decorator registry** — add custom adapters without touching core code
- **Strategy pattern** — swap LoRA / QLoRA / SFT / DPO / ORPO / IA³ independently of the model
- **Gradient accumulation, AMP (fp16/bf16), grad clipping** — production training out of the box
- **Checkpoint + resume** — save every N epochs, resume from any checkpoint
- **Early stopping** — configurable patience and delta
- **WandB integration** — `--wandb-project` flag, no code changes needed
- **Inference CLI** — `unify-train infer` runs generation on a saved model
- **297 tests, GitHub Actions CI**

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Install](#install)
- [Usage](#usage)
- [Example Configs](#example-configs)
- [Built-in Adapters](#built-in-adapters)
- [Extending](#extending)
- [Contributing](#contributing)
- [License](#license)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Core framework | Python 3.10+, PyTorch 2.0+ |
| LLM adapters | HuggingFace Transformers, PEFT |
| Config validation | Pydantic v2 |
| CLI | Click |
| Experiment tracking | Weights & Biases (optional) |
| Metrics | sacrebleu, rouge-score, scikit-learn (optional) |
| Testing | pytest, pytest-mock |
| CI | GitHub Actions + uv |

---

## Architecture

```
unify-train/
├── core/
│   ├── base/          # Abstract base classes (BaseModel, BaseStrategy, …)
│   ├── config/        # Pydantic schema + YAML loader
│   ├── registry/      # Decorator-based component registry
│   ├── trainer/       # Training loop, checkpoint utils
│   ├── hook/          # Hook system (checkpoint, CSV, early stopping, WandB)
│   └── cli/           # Click CLI (train, infer, list-adapters)
└── adapters/
    ├── llm/           # LLM adapters (models, strategies, datasets, evaluators)
    └── vision/        # Vision adapters (models, strategies, datasets, evaluators)
```

**Data flow:**

```
YAML config → Pydantic validation → Registry lookup → Trainer
                                                         │
                                    ┌────────────────────┤
                                    ▼                    ▼
                              Strategy.setup()      DataLoader
                                    │                    │
                              optimizer, scheduler   batch loop
                                    │                    │
                                    └──→ training_step ──┘
                                              │
                                         Hook events
                                    (after_step, after_epoch, …)
```

---

## Install

```bash
# Core
pip install -e ".[dev]"

# Optional — install only what you need
pip install -e ".[qlora]"               # QLoRA 4-bit quantization
pip install -e ".[generation-eval]"     # BLEU + ROUGE-L evaluator
pip install -e ".[classification-eval]" # F1 / precision / recall evaluator
pip install -e ".[wandb]"               # WandB experiment tracking
pip install -e ".[all-extras]"          # Everything above
```

**Requirements:** Python 3.10+, PyTorch 2.0+

---

## Usage

```bash
# Train
unify-train train --config configs/my_run.yaml

# Train with checkpoint every 2 epochs
unify-train train --config configs/my_run.yaml --checkpoint-every 2

# Resume from a checkpoint
unify-train train --config configs/my_run.yaml --resume outputs/my_run/checkpoints/epoch_2

# Train with early stopping
unify-train train --config configs/my_run.yaml --early-stopping-patience 3

# Train with WandB logging (requires .[wandb])
unify-train train --config configs/my_run.yaml --wandb-project my-project --wandb-run run-1

# Run inference on a trained model
unify-train infer --config configs/my_run.yaml --prompt "What is the capital of France?"
unify-train infer --config configs/my_run.yaml --prompt "Hello" --max-new-tokens 200 --temperature 0.8
unify-train infer --config configs/my_run.yaml --prompt "Hello" --checkpoint outputs/my_run/checkpoints/epoch_2

# List all registered adapters
unify-train list-adapters
```

---

## Example Configs

### LLaMA-3.2-1B + QLoRA + instruction data

```yaml
model:
  name: hf_causal_lm
  params:
    pretrained: meta-llama/Llama-3.2-1B
    load_in_4bit: true

dataset:
  name: instruction
  params:
    path: tatsu-lab/alpaca      # HF dataset ID or local .jsonl path
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

### GPT-2 + LoRA (minimal, CPU-friendly)

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
  device: cpu

output_dir: outputs/gpt2-lora
```

More examples in [`configs/`](configs/).

---

## Built-in Adapters

| Type | Name | Description |
|---|---|---|
| **Models** | `hf_causal_lm` | Any HuggingFace causal LM (LLaMA, Mistral, Phi, GPT-2, …) |
| | `resnet18` | ResNet-18 image classifier |
| **Strategies** | `lora` | Low-Rank Adaptation |
| | `qlora` | 4-bit NF4 quantization + LoRA. Requires `.[qlora]` + CUDA |
| | `sft` | Supervised fine-tuning (full fine-tuning) |
| | `ia3` | IA³ — fewer trainable params than LoRA |
| | `dpo` | Direct Preference Optimization |
| | `orpo` | Odds Ratio Preference Optimization (no reference model) |
| | `supervised` | Standard supervised training for vision |
| **Datasets** | `instruction` | Alpaca / ShareGPT formats with prompt masking |
| | `text_clm` | In-memory text for causal LM |
| | `hf_text` | Any HuggingFace text dataset |
| | `hf_dpo` | Preference pairs for DPO / ORPO |
| | `synthetic_image` | Random images for vision testing |
| **Evaluators** | `perplexity` | Language model perplexity |
| | `generation` | BLEU + ROUGE-L. Requires `.[generation-eval]` |
| | `classification` | F1, precision, recall, exact-match. Requires `.[classification-eval]` |
| | `accuracy` | Top-1 accuracy for image classification |
| **Hooks** | `CheckpointHook` | Save model every N epochs |
| | `CSVLoggerHook` | Log metrics to CSV |
| | `EarlyStoppingHook` | Stop when loss plateaus |
| | `WandbLoggerHook` | Log to W&B. Requires `.[wandb]` |

---

## Extending

Register a custom component with a decorator — no changes to core required:

```python
# my_adapter.py
from core.base.model import BaseModel
from core.base.io_types import TEXT_TO_TEXT
from core.registry import MODELS

@MODELS.register("my_model")
class MyModel(BaseModel):
    io_type = TEXT_TO_TEXT

    def build(self, config): ...
    def forward(self, batch): ...
    def save(self, path): ...
    def load(self, path): ...

    @property
    def raw_model(self): ...
```

Same pattern for `STRATEGIES`, `DATASETS`, `EVALUATORS`. Then reference by name in your YAML:

```yaml
model:
  name: my_model
```

---

## Contributing

Contributions are welcome. To get started:

```bash
git clone https://github.com/tdm291104/unify-train.git
cd unify-train
pip install -e ".[dev]"
uv run pytest        # run the test suite
```

Please:
- Add tests for any new adapter (see `tests/adapters/llm/` for examples)
- Follow the existing registry pattern — one file per adapter, registered via decorator
- Run `uv run pytest -x -q` before opening a PR

---

## License

[MIT](LICENSE)
