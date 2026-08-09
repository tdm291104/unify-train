import csv
import os
from core.hook.hooks import HookContext
from core.hook.csv_logger import CSVLoggerHook


def _ctx(epoch: int = 0, step: int = 1, loss: float = 1.0, metrics: dict = None) -> HookContext:
    ctx = HookContext(epoch=epoch, step=step, loss=loss, metrics=metrics or {})
    return ctx


def test_creates_file_on_first_call(tmp_path):
    path = str(tmp_path / "log.csv")
    hook = CSVLoggerHook(path)
    hook(_ctx())
    assert os.path.exists(path)


def test_header_written_correctly(tmp_path):
    path = str(tmp_path / "log.csv")
    hook = CSVLoggerHook(path)
    hook(_ctx())
    with open(path) as f:
        header = f.readline().strip().split(",")
    assert header == ["epoch", "step", "loss"]


def test_row_values_correct(tmp_path):
    path = str(tmp_path / "log.csv")
    hook = CSVLoggerHook(path)
    hook(_ctx(epoch=2, step=5, loss=0.42))
    with open(path) as f:
        reader = list(csv.DictReader(f))
    assert reader[0]["epoch"] == "2"
    assert reader[0]["step"] == "5"
    assert abs(float(reader[0]["loss"]) - 0.42) < 1e-6


def test_appends_multiple_rows(tmp_path):
    path = str(tmp_path / "log.csv")
    hook = CSVLoggerHook(path)
    for i in range(3):
        hook(_ctx(step=i, loss=float(i)))
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


def test_extra_metrics_included_as_columns(tmp_path):
    path = str(tmp_path / "log.csv")
    hook = CSVLoggerHook(path)
    hook(_ctx(metrics={"accuracy": 0.9}))
    with open(path) as f:
        reader = list(csv.DictReader(f))
    assert "accuracy" in reader[0]
    assert abs(float(reader[0]["accuracy"]) - 0.9) < 1e-6


def test_creates_parent_directory(tmp_path):
    path = str(tmp_path / "subdir" / "log.csv")
    hook = CSVLoggerHook(path)
    hook(_ctx())
    assert os.path.exists(path)
