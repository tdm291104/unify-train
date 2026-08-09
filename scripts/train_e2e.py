"""End-to-end training script for unify-train."""
import argparse
import logging

import adapters.llm  # noqa: F401 — triggers all adapter registrations

from core.config.loader import load_config
from core.hook.hooks import HookManager
from core.trainer.trainer import build_trainer_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="unify-train e2e training")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info("Loaded config: model=%s strategy=%s epochs=%d", cfg.model.name, cfg.strategy.name, cfg.train.max_epochs)

    hooks = HookManager()

    def on_before_train(ctx):
        log.info("Training start")

    def on_after_epoch(ctx):
        log.info("Epoch %d done | loss=%.4f", ctx.epoch, ctx.loss)

    def on_after_step(ctx):
        if ctx.step % cfg.train.log_every_n_steps == 0:
            log.info("Step %d | loss=%.4f", ctx.step, ctx.loss)

    def on_after_train(ctx):
        log.info("Training complete")

    hooks.register("before_train", on_before_train)
    hooks.register("after_epoch", on_after_epoch)
    hooks.register("after_step", on_after_step)
    hooks.register("after_train", on_after_train)

    trainer = build_trainer_from_config(cfg, hooks=hooks)
    trainer.run()
    log.info("Model saved to %s", cfg.output_dir)


if __name__ == "__main__":
    main()
