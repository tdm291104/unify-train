from core.hook.hooks import HookContext


class WandbLoggerHook:
    """Logs training metrics to Weights & Biases. Requires: pip install wandb"""

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self._project = project
        self._run_name = run_name
        self._config = config or {}
        self._tags = tags or []
        self._wandb = None

    def on_before_train(self, ctx: HookContext) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("WandbLoggerHook requires wandb: pip install wandb") from exc
        self._wandb = wandb
        wandb.init(
            project=self._project,
            name=self._run_name,
            config=self._config,
            tags=self._tags,
        )

    def on_after_step(self, ctx: HookContext) -> None:
        if self._wandb is None:
            return
        payload = {"train/loss": ctx.loss, "step": ctx.step}
        for k, v in ctx.metrics.items():
            if k != "loss":
                payload[f"train/{k}"] = v
        self._wandb.log(payload, step=ctx.step)

    def on_after_eval(self, ctx: HookContext) -> None:
        if self._wandb is None:
            return
        payload = {f"eval/{k}": v for k, v in ctx.metrics.items()}
        payload["step"] = ctx.step
        self._wandb.log(payload, step=ctx.step)

    def on_after_train(self, ctx: HookContext) -> None:
        if self._wandb is not None:
            self._wandb.finish()
