import torch
from core.base.io_types import IMAGE_TO_CATEGORY


def test_synthetic_image_registers():
    import adapters.vision.datasets.synthetic
    from core.registry import DATASETS
    assert "synthetic_image" in DATASETS.list_all()


def test_synthetic_image_io_type():
    import adapters.vision.datasets.synthetic
    from core.registry import DATASETS
    assert DATASETS.get("synthetic_image").io_type == IMAGE_TO_CATEGORY


def test_len_matches_size_param():
    from adapters.vision.datasets.synthetic import SyntheticImageDataset
    ds = SyntheticImageDataset({"size": 20, "num_classes": 5})
    assert len(ds) == 20


def test_getitem_returns_correct_shape():
    from adapters.vision.datasets.synthetic import SyntheticImageDataset
    ds = SyntheticImageDataset({"size": 10, "num_classes": 3, "channels": 3, "image_size": 64})
    sample = ds[0]
    assert sample["pixel_values"].shape == (3, 64, 64)
    assert sample["labels"].ndim == 0


def test_labels_within_range():
    from adapters.vision.datasets.synthetic import SyntheticImageDataset
    ds = SyntheticImageDataset({"size": 50, "num_classes": 4})
    for i in range(len(ds)):
        label = ds[i]["labels"].item()
        assert 0 <= label < 4


def test_collate_fn_stacks_correctly():
    from adapters.vision.datasets.synthetic import SyntheticImageDataset
    ds = SyntheticImageDataset({"size": 10, "num_classes": 3, "channels": 3, "image_size": 64})
    samples = [ds[i] for i in range(4)]
    batch = ds.collate_fn(samples)
    assert batch["pixel_values"].shape == (4, 3, 64, 64)
    assert batch["labels"].shape == (4,)


def test_reproducible_with_seed():
    from adapters.vision.datasets.synthetic import SyntheticImageDataset
    ds1 = SyntheticImageDataset({"size": 5, "seed": 7})
    ds2 = SyntheticImageDataset({"size": 5, "seed": 7})
    assert torch.equal(ds1[0]["pixel_values"], ds2[0]["pixel_values"])
    assert torch.equal(ds1[0]["labels"], ds2[0]["labels"])
