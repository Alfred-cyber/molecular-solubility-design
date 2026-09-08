import pandas as pd
import pytest

from molecular_engineering.data_loader import (
    load_delaney_splits,
    DatasetLoadError,
    _to_dataframe,
    _clean,
)


class FakeHFSplit:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self):
        return self._df.copy()


class FakeHFDatasetDict(dict):
    pass


def make_fake_raw():
    train = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(=O)O", None, "CCO"],
        "target": [0.1, -2.1, -0.2, -1.0, 0.1],
    })
    validation = pd.DataFrame({
        "smiles": ["CCN", "CCC"],
        "target": [-0.5, -1.8],
    })
    test = pd.DataFrame({
        "smiles": ["CCCl", "invalid but not caught here"],
        "target": [-1.2, None],
    })
    return FakeHFDatasetDict(
        train=FakeHFSplit(train),
        validation=FakeHFSplit(validation),
        test=FakeHFSplit(test),
    )


def test_load_delaney_splits_success(monkeypatch):
    import molecular_engineering.data_loader as data_loader_module

    def fake_load_dataset(name):
        assert name == "zpn/delaney"
        return make_fake_raw()

    monkeypatch.setattr(
        "datasets.load_dataset", fake_load_dataset
    )

    splits = load_delaney_splits()
    assert set(splits.keys()) == {"train", "validation", "test"}
    assert len(splits["train"]) == 3
    assert len(splits["test"]) == 1
    assert splits["test"]["smiles"].iloc[0] == "CCCl"


def test_load_delaney_splits_missing_train_raises(monkeypatch):
    def fake_load_dataset(name):
        return FakeHFDatasetDict(test=FakeHFSplit(pd.DataFrame({"smiles": ["CCO"], "target": [0.1]})))

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    with pytest.raises(DatasetLoadError):
        load_delaney_splits()


def test_load_delaney_splits_download_failure_raises(monkeypatch):
    def fake_load_dataset(name):
        raise ConnectionError("no network")

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    with pytest.raises(DatasetLoadError):
        load_delaney_splits()


def test_to_dataframe_missing_columns_raises():
    split = FakeHFSplit(pd.DataFrame({"smiles": ["CCO"]}))
    with pytest.raises(DatasetLoadError):
        _to_dataframe(split, "smiles", "target")


def test_clean_removes_missing_and_duplicates():
    df = pd.DataFrame({
        "smiles": ["CCO", None, "CCO", "c1ccccc1"],
        "target": [0.1, -1.0, 0.1, None],
    })
    cleaned = _clean(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["smiles"] == "CCO"
