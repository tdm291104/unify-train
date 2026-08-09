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
def train(
    config: str,
    checkpoint_every: int,
    no_checkpoint: bool,
    early_stopping_patience: int,
    early_stopping_delta: float,
    resume: str | None,
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

    if resume:
        log.info("Resuming from checkpoint: %s", resume)

    trainer = build_trainer_from_config(cfg, hooks=hooks, resume=resume)
    trainer.run()
    log.info("Model saved to %s", cfg.output_dir)


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
