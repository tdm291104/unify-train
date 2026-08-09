import pytest
import torch
from unittest.mock import MagicMock, patch
from core.base.io_types import IMAGE_TO_CATEGORY


def test_resnet18_registers():
    import adapters.vision.models.resnet18
    from core.registry import MODELS
    assert "resnet18" in MODELS.list_all()


def test_resnet18_io_type():
    import adapters.vision.models.resnet18
    from core.registry import MODELS
    assert MODELS.get("resnet18").io_type == IMAGE_TO_CATEGORY


def test_resnet18_build_sets_num_classes():
    from adapters.vision.models.resnet18 import ResNet18Model
    m = ResNet18Model()
    m.build({"num_classes": 5})
    assert m.raw_model.fc.out_features == 5


def test_resnet18_build_default_num_classes():
    from adapters.vision.models.resnet18 import ResNet18Model
    m = ResNet18Model()
    m.build({})
    assert m.raw_model.fc.out_features == 10


def test_resnet18_forward_returns_logits():
    from adapters.vision.models.resnet18 import ResNet18Model
    m = ResNet18Model()
    m.build({"num_classes": 3})
    batch = {"pixel_values": torch.randn(2, 3, 64, 64)}
    out = m.forward(batch)
    assert "logits" in out
    assert out["logits"].shape == (2, 3)


def test_resnet18_forward_with_labels_returns_loss():
    from adapters.vision.models.resnet18 import ResNet18Model
    m = ResNet18Model()
    m.build({"num_classes": 3})
    batch = {
        "pixel_values": torch.randn(2, 3, 64, 64),
        "labels": torch.tensor([0, 2]),
    }
    out = m.forward(batch)
    assert "loss" in out
    assert out["loss"].ndim == 0


def test_resnet18_forward_without_labels_no_loss():
    from adapters.vision.models.resnet18 import ResNet18Model
    m = ResNet18Model()
    m.build({"num_classes": 3})
    m.raw_model.eval()
    with torch.no_grad():
        out = m.forward({"pixel_values": torch.randn(1, 3, 64, 64)})
    assert "loss" not in out


def test_resnet18_save_load_roundtrip(tmp_path):
    from adapters.vision.models.resnet18 import ResNet18Model
    m1 = ResNet18Model()
    m1.build({"num_classes": 4})
    m1.save(str(tmp_path))

    m2 = ResNet18Model()
    m2.build({"num_classes": 4})
    m2.load(str(tmp_path))

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        m1.raw_model.eval()
        m2.raw_model.eval()
        out1 = m1.forward({"pixel_values": x})
        out2 = m2.forward({"pixel_values": x})
    assert torch.allclose(out1["logits"], out2["logits"])
