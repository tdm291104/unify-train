import logging
import os
import sys

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """unify-train — standardized ML training framework."""


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--checkpoint-every", default=1, show_default=True, help="Save checkpoint every N epochs.")
@click.option("--no-checkpoint", is_flag=True, help="Disable checkpointing.")
@click.option("--early-stopping-patience", default=0, show_default=True,
              help="Stop after N epochs with no improvement (0 = disabled).")
@click.option("--early-stopping-delta", default=0.0, show_default=True,
              help="Minimum loss improvement to count as progress.")
@click.option("--resume", default=None, type=click.Path(exists=True, file_okay=False),
              help="Resume from a checkpoint directory.")
@click.option("--wandb-project", default=None, type=str,
              help="Log to W&B project (requires wandb installed).")
@click.option("--wandb-run", default=None, type=str,
              help="W&B run name (optional, defaults to W&B auto-name).")
def train(
    config: str,
    checkpoint_every: int,
    no_checkpoint: bool,
    early_stopping_patience: int,
    early_stopping_delta: float,
    resume: str | None,
    wandb_project: str | None,
    wandb_run: str | None,
) -> None:
    """Run training from a YAML config file."""
    import adapters.llm  # noqa: F401
    import adapters.vision  # noqa: F401
    from core.config.loader import load_config
    from core.hook import HookManager, CheckpointHook, CSVLoggerHook, EarlyStoppingHook
    from core.trainer.trainer import build_trainer_from_config

    cfg = load_config(config)
    log.info("model=%s  strategy=%s  epochs=%d", cfg.model.name, cfg.strategy.name, cfg.train.max_epochs)

    hooks = HookManager()

    hooks.register("before_train", lambda ctx: log.info("Training start"))
    hooks.register("after_epoch", lambda ctx: log.info("Epoch %d done | loss=%.4f", ctx.epoch, ctx.loss))
    hooks.register("after_step", lambda ctx: (
        log.info("Step %d | loss=%.4f", ctx.step, ctx.loss)
        if ctx.step % cfg.train.log_every_n_steps == 0 else None
    ))
    hooks.register("after_train", lambda ctx: log.info("Training complete"))

    if not no_checkpoint:
        ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
        hooks.register("after_epoch", CheckpointHook(ckpt_dir, every=checkpoint_every))

    hooks.register("after_step", CSVLoggerHook(os.path.join(cfg.output_dir, "train_log.csv")))

    if early_stopping_patience > 0:
        hooks.register("after_epoch", EarlyStoppingHook(
            patience=early_stopping_patience,
            min_delta=early_stopping_delta,
        ))

    if wandb_project:
        from core.hook import WandbLoggerHook
        wandb_cfg = {
            "model": cfg.model.name,
            "strategy": cfg.strategy.name,
            "dataset": cfg.dataset.name,
            "max_epochs": cfg.train.max_epochs,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.strategy.params.get("lr"),
        }
        wandb_hook = WandbLoggerHook(
            project=wandb_project,
            run_name=wandb_run,
            config=wandb_cfg,
        )
        hooks.register("before_train", wandb_hook.on_before_train)
        hooks.register("after_step", wandb_hook.on_after_step)
        hooks.register("after_eval", wandb_hook.on_after_eval)
        hooks.register("after_train", wandb_hook.on_after_train)

    if resume:
        log.info("Resuming from checkpoint: %s", resume)

    trainer = build_trainer_from_config(cfg, hooks=hooks, resume=resume)
    trainer.run()
    log.info("Model saved to %s", cfg.output_dir)


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--prompt", required=True, help="Input text prompt.")
@click.option("--max-new-tokens", default=200, show_default=True, help="Maximum tokens to generate.")
@click.option("--temperature", default=1.0, show_default=True, help="Sampling temperature (1.0 = greedy-ish).")
@click.option("--checkpoint", default=None, type=click.Path(exists=True, file_okay=False),
              help="Load weights from this checkpoint dir instead of output_dir.")
def infer(config: str, prompt: str, max_new_tokens: int, temperature: float, checkpoint: str | None) -> None:
    """Run inference using a trained model."""
    import adapters.llm  # noqa: F401
    import adapters.vision  # noqa: F401
    from core.config.loader import load_config
    from core.registry import MODELS

    if temperature <= 0.0:
        raise click.ClickException("--temperature must be > 0.0")

    cfg = load_config(config)
    model_dir = checkpoint or cfg.output_dir

    if not os.path.isdir(model_dir):
        raise click.ClickException(
            f"Model directory '{model_dir}' does not exist. "
            "Train the model first or pass --checkpoint."
        )

    model_cls = MODELS.get(cfg.model.name)
    model = model_cls()
    model.build(cfg.model.params)
    model.load(model_dir)

    # Tokenize the prompt using the model's tokenizer (if available)
    if not hasattr(model, "tokenizer") or model.tokenizer is None:
        raise click.ClickException(
            f"Model '{cfg.model.name}' does not expose a tokenizer. "
            "Inference requires a model with a tokenizer property (e.g. hf_causal_lm)."
        )

    tokenizer = model.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Move to CPU (inference CLI doesn't assume CUDA)
    model.raw_model.eval()

    import torch
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature != 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    generated = output_ids[0][input_ids.shape[-1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    click.echo(result)


@main.command("list-adapters")
def list_adapters() -> None:
    """List all registered models, datasets, strategies, and evaluators."""
    import adapters.llm  # noqa: F401
    import adapters.vision  # noqa: F401
    from core.registry import MODELS, DATASETS, STRATEGIES, EVALUATORS

    sections = [
        ("Models", MODELS),
        ("Datasets", DATASETS),
        ("Strategies", STRATEGIES),
        ("Evaluators", EVALUATORS),
    ]
    for title, registry in sections:
        names = registry.list_all()
        click.echo(f"\n{title} ({len(names)}):")
        for name in sorted(names):
            click.echo(f"  • {name}")
    click.echo()
