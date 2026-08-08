import pytest
from core.base.io_types import DataType, IOType, TEXT_TO_TEXT, IMAGE_TO_CATEGORY, AUDIO_TO_TEXT


def test_iotype_equality():
    assert IOType(DataType.TEXT, DataType.TEXT) == IOType(DataType.TEXT, DataType.TEXT)


def test_iotype_inequality():
    assert IOType(DataType.TEXT, DataType.TEXT) != IOType(DataType.IMAGE, DataType.CATEGORY)


def test_iotype_hashable():
    io = IOType(DataType.TEXT, DataType.TEXT)
    assert io in {io}


def test_iotype_as_dict_key():
    d = {IOType(DataType.TEXT, DataType.TEXT): "text_task"}
    assert d[IOType(DataType.TEXT, DataType.TEXT)] == "text_task"


def test_iotype_str():
    assert str(IOType(DataType.TEXT, DataType.TEXT)) == "text->text"
    assert str(IOType(DataType.IMAGE, DataType.CATEGORY)) == "image->category"


def test_predefined_shortcuts():
    assert TEXT_TO_TEXT == IOType(DataType.TEXT, DataType.TEXT)
    assert IMAGE_TO_CATEGORY == IOType(DataType.IMAGE, DataType.CATEGORY)
    assert AUDIO_TO_TEXT == IOType(DataType.AUDIO, DataType.TEXT)


def test_datatype_str_values():
    assert DataType.TEXT.value == "text"
    assert DataType.IMAGE.value == "image"
    assert DataType.AUDIO.value == "audio"
    assert DataType.CATEGORY.value == "category"
    assert DataType.TABULAR.value == "tabular"
