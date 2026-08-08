import pytest
from core.base.io_types import IOType, DataType, TEXT_TO_TEXT
from core.base.model import BaseModel
from core.base.dataset import BaseDataset
from core.base.strategy import BaseStrategy
from core.base.evaluator import BaseEvaluator


# --- Minimal concrete implementations for testing ---

class ConcreteModel(BaseModel):
    io_type = TEXT_TO_TEXT
    def build(self, config): pass
    def forward(self, batch): return {"loss": 0.0}
    def save(self, path): pass
    def load(self, path): pass


class ConcreteDataset(BaseDataset):
    io_type = TEXT_TO_TEXT
    def __len__(self): return 0
    def __getitem__(self, idx): return {}
    def collate_fn(self, samples): return {}


class ConcreteStrategy(BaseStrategy):
    compatible_io_types = [TEXT_TO_TEXT]
    compatible_tasks: list[str] = []
    def setup(self, model, config): return model
    def configure_optimizers(self, model, config): return None
    def training_step(self, model, batch, optimizer, step): return {"loss": 0.0}
    def teardown(self, model): pass


class ConcreteEvaluator(BaseEvaluator):
    io_type = TEXT_TO_TEXT
    def compute(self, predictions, references): return {"metric": 0.0}


# --- ABC enforcement ---

def test_cannot_instantiate_base_model():
    with pytest.raises(TypeError):
        BaseModel()


def test_cannot_instantiate_base_dataset():
    with pytest.raises(TypeError):
        BaseDataset()


def test_cannot_instantiate_base_strategy():
    with pytest.raises(TypeError):
        BaseStrategy()


def test_cannot_instantiate_base_evaluator():
    with pytest.raises(TypeError):
        BaseEvaluator()


# --- Concrete instantiation ---

def test_concrete_model_io_type():
    assert ConcreteModel.io_type == TEXT_TO_TEXT


def test_concrete_dataset_io_type():
    assert ConcreteDataset.io_type == TEXT_TO_TEXT


def test_concrete_evaluator_compute():
    result = ConcreteEvaluator().compute([0.5], [0.5])
    assert "metric" in result


# --- BaseStrategy.validate() default logic ---

def test_strategy_validate_passes_for_compatible_io_type():
    ConcreteStrategy().validate(TEXT_TO_TEXT, None)  # must not raise


def test_strategy_validate_fails_for_incompatible_io_type():
    s = ConcreteStrategy()
    wrong = IOType(DataType.IMAGE, DataType.CATEGORY)
    with pytest.raises(ValueError, match="io_type"):
        s.validate(wrong, None)


def test_strategy_validate_fails_for_incompatible_task():
    class TaskLimited(BaseStrategy):
        compatible_io_types = [TEXT_TO_TEXT]
        compatible_tasks = ["translation"]
        def setup(self, m, c): return m
        def configure_optimizers(self, m, c): return None
        def training_step(self, m, b, o, s): return {"loss": 0.0}
        def teardown(self, m): pass

    with pytest.raises(ValueError, match="task"):
        TaskLimited().validate(TEXT_TO_TEXT, "summarization")


def test_strategy_validate_passes_with_empty_compatible_tasks():
    # compatible_tasks=[] means no restriction on task
    ConcreteStrategy().validate(TEXT_TO_TEXT, "any_task_name")  # must not raise


def test_strategy_validate_passes_with_empty_compatible_io_types():
    class Unrestricted(BaseStrategy):
        compatible_io_types: list = []
        compatible_tasks: list[str] = []
        def setup(self, m, c): return m
        def configure_optimizers(self, m, c): return None
        def training_step(self, m, b, o, s): return {"loss": 0.0}
        def teardown(self, m): pass

    Unrestricted().validate(IOType(DataType.IMAGE, DataType.CATEGORY), "anything")


def test_model_without_io_type_raises_on_class_definition():
    with pytest.raises(TypeError, match="io_type"):
        class BadModel(BaseModel):
            def build(self, config): pass
            def forward(self, batch): return {}
            def save(self, path): pass
            def load(self, path): pass
