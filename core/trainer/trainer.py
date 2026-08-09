import os
import torch
from torch.utils.data import DataLoader
from core.base.model import BaseModel
from core.base.dataset import BaseDataset
from core.base.strategy import BaseStrategy
from core.base.evaluator import BaseEvaluator
from core.base.io_types import IOType, DataType
from core.config.schema import UnifyTrainConfig
from core.hook.hooks import HookManager, HookContext
from core.hook.early_stopping import StopTraining


class Trainer:

    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        strategy: BaseStrategy,
        config: UnifyTrainConfig,
        evaluator: BaseEvaluator | None = None,
        eval_dataset: BaseDataset | None = None,
        hooks: HookManager | None = None,
        start_epoch: int = 0,
        start_step: int = 0,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._strategy = strategy
        self._config = config
        self._evaluator = evaluator
        self._eval_dataset = eval_dataset
        self._hooks = hooks or HookManager()
        self._start_epoch = start_epoch
        self._start_step = start_step
        self._io_type = IOType(
            input=DataType(config.io_type.input),
            output=DataType(config.io_type.output),
        )

    def _run_eval(self, model: BaseModel, epoch: int, ctx: HookContext) -> None:
        if self._evaluator is None or self._eval_dataset is None:
            return
        if (epoch + 1) % self._config.eval.eval_every_n_epochs != 0:
            return

        model.raw_model.eval()
        loader = DataLoader(
            self._eval_dataset,
            batch_size=self._config.train.batch_size,
            collate_fn=self._eval_dataset.collate_fn,
            shuffle=False,
        )
        all_preds: list = []
        all_refs: list = []
        with torch.no_grad():
            for batch in loader:
                out = model.forward(batch)
                preds, refs = self._evaluator.extract(out, batch)
                all_preds.extend(preds)
                all_refs.extend(refs)

        eval_metrics = self._evaluator.compute(all_preds, all_refs)
        ctx.metrics = {**ctx.metrics, **eval_metrics}
        model.raw_model.train()
        self._hooks.fire("after_eval", ctx)

    def run(self) -> None:
        cfg = self._config

        self._strategy.validate(self._io_type, cfg.task)

        model = self._strategy.setup(self._model, cfg.strategy.params)
        optimizer = self._strategy.configure_optimizers(model, cfg.strategy.params)
        scheduler = self._strategy.configure_scheduler(optimizer, cfg.strategy.params)

        loader = DataLoader(
            self._dataset,
            batch_size=cfg.train.batch_size,
            collate_fn=self._dataset.collate_fn,
            shuffle=True,
        )

        ctx = HookContext(model=model, config=cfg)
        self._hooks.fire("before_train", ctx)

        global_step = self._start_step
        try:
            for epoch in range(self._start_epoch, cfg.train.max_epochs):
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
                self._run_eval(model, epoch, ctx)
                if scheduler is not None:
                    scheduler.step()
        except StopTraining:
            pass

        self._hooks.fire("after_train", ctx)
        self._strategy.teardown(model)

        os.makedirs(cfg.output_dir, exist_ok=True)
        model.save(cfg.output_dir)


def build_trainer_from_config(
    config: UnifyTrainConfig,
    hooks: HookManager | None = None,
    resume: str | None = None,
) -> Trainer:
    from core.registry import MODELS, DATASETS, STRATEGIES, EVALUATORS
    from core.trainer.checkpoint_utils import load_checkpoint

    io_type = IOType(
        input=DataType(config.io_type.input),
        output=DataType(config.io_type.output),
    )

    model_cls = MODELS.get(config.model.name)
    model = model_cls()
    model.build({**config.model.params, "io_type": io_type})

    start_epoch, start_step = 0, 0
    if resume:
        start_epoch, start_step = load_checkpoint(resume, model)

    dataset_cls = DATASETS.get(config.dataset.name)
    dataset = dataset_cls(config.dataset.params)

    strategy_cls = STRATEGIES.get(config.strategy.name)
    strategy = strategy_cls()

    evaluator = None
    if config.evaluator:
        evaluator_cls = EVALUATORS.get(config.evaluator.name)
        evaluator = evaluator_cls()

    eval_dataset = None
    if config.eval_dataset:
        eval_dataset_cls = DATASETS.get(config.eval_dataset.name)
        eval_dataset = eval_dataset_cls(config.eval_dataset.params)

    return Trainer(
        model=model,
        dataset=dataset,
        strategy=strategy,
        config=config,
        evaluator=evaluator,
        eval_dataset=eval_dataset,
        hooks=hooks,
        start_epoch=start_epoch,
        start_step=start_step,
    )
