import os
from torch.utils.data import DataLoader
from core.base.model import BaseModel
from core.base.dataset import BaseDataset
from core.base.strategy import BaseStrategy
from core.base.evaluator import BaseEvaluator
from core.base.io_types import IOType, DataType
from core.config.schema import UnifyTrainConfig
from core.hook.hooks import HookManager, HookContext


class Trainer:

    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        strategy: BaseStrategy,
        config: UnifyTrainConfig,
        evaluator: BaseEvaluator | None = None,
        hooks: HookManager | None = None,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._strategy = strategy
        self._config = config
        self._evaluator = evaluator
        self._hooks = hooks or HookManager()
        self._io_type = IOType(
            input=DataType(config.io_type.input),
            output=DataType(config.io_type.output),
        )

    def run(self) -> None:
        cfg = self._config

        self._strategy.validate(self._io_type, cfg.task)

        model = self._strategy.setup(self._model, cfg.strategy.params)
        optimizer = self._strategy.configure_optimizers(model, cfg.strategy.params)

        loader = DataLoader(
            self._dataset,
            batch_size=cfg.train.batch_size,
            collate_fn=self._dataset.collate_fn,
            shuffle=True,
        )

        ctx = HookContext(model=model, config=cfg)
        self._hooks.fire("before_train", ctx)

        global_step = 0
        for epoch in range(cfg.train.max_epochs):
            ctx.epoch = epoch
            self._hooks.fire("before_epoch", ctx)

            for batch in loader:
                metrics = self._strategy.training_step(
                    model, batch, optimizer, step=global_step
                )
                if "loss" not in metrics:
                    raise ValueError(
                        f"{type(self._strategy).__name__}.training_step must return a dict "
                        f"with 'loss' key, got keys: {list(metrics)}"
                    )
                global_step += 1
                ctx.step = global_step
                ctx.loss = metrics["loss"]
                ctx.metrics = metrics
                self._hooks.fire("after_step", ctx)

            self._hooks.fire("after_epoch", ctx)

        self._hooks.fire("after_train", ctx)
        self._strategy.teardown(model)

        os.makedirs(cfg.output_dir, exist_ok=True)
        model.save(cfg.output_dir)


def build_trainer_from_config(
    config: UnifyTrainConfig,
    hooks: HookManager | None = None,
) -> Trainer:
    from core.registry import MODELS, DATASETS, STRATEGIES, EVALUATORS

    io_type = IOType(
        input=DataType(config.io_type.input),
        output=DataType(config.io_type.output),
    )

    model_cls = MODELS.get(config.model.name)
    model = model_cls()
    model.build({**config.model.params, "io_type": io_type})

    dataset_cls = DATASETS.get(config.dataset.name)
    dataset = dataset_cls(config.dataset.params)

    strategy_cls = STRATEGIES.get(config.strategy.name)
    strategy = strategy_cls()

    evaluator = None
    if config.evaluator:
        evaluator_cls = EVALUATORS.get(config.evaluator.name)
        evaluator = evaluator_cls()

    return Trainer(
        model=model,
        dataset=dataset,
        strategy=strategy,
        config=config,
        evaluator=evaluator,
        hooks=hooks,
    )
