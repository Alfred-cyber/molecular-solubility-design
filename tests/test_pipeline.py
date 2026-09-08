import pandas as pd
import numpy as np

from molecular_engineering.config import Config
from molecular_engineering.pipeline import run_training_pipeline

SMILES_POOL = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "CCCl", "CCCC", "OCC",
    "CC(C)O", "c1ccccc1O", "CCOCC", "CCCCCC", "CCCCO", "c1ccncc1", "CC(=O)OC",
    "CCCCCCCC", "c1ccc(O)cc1", "CC(C)(C)O", "CCCCCl", "OCCO",
    "c1ccc2ccccc2c1", "CC(C)CC(=O)O", "CCCCCCCCCC", "c1ccsc1", "CCOC(=O)C",
]


class FakeHFSplit:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df.copy()


class FakeHFDatasetDict(dict):
    pass


def make_fake_raw(seed=0):
    rng = np.random.default_rng(seed)
    targets = rng.normal(-2.0, 1.0, size=len(SMILES_POOL))
    df = pd.DataFrame({"smiles": SMILES_POOL, "target": targets})
    train = df.iloc[:16].reset_index(drop=True)
    validation = df.iloc[16:20].reset_index(drop=True)
    test = df.iloc[20:].reset_index(drop=True)
    return FakeHFDatasetDict(
        train=FakeHFSplit(train),
        validation=FakeHFSplit(validation),
        test=FakeHFSplit(test),
    )


def test_run_training_pipeline_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr("datasets.load_dataset", lambda name: make_fake_raw())

    config = Config(
        models_dir=str(tmp_path / "models"),
        results_dir=str(tmp_path / "results"),
        random_seed=42,
    )

    result = run_training_pipeline(config)
    summary = result["summary"]

    assert summary["selected_model"] in {"linear_regression", "random_forest", "gradient_boosting"}
    assert "MAE" in summary["test_metrics"]
    assert (tmp_path / "results" / "training_summary.json").exists()
    assert config.model_path.exists()


def test_run_training_pipeline_is_reproducible_with_same_seed(monkeypatch, tmp_path):
    monkeypatch.setattr("datasets.load_dataset", lambda name: make_fake_raw())

    config_a = Config(models_dir=str(tmp_path / "a"), results_dir=str(tmp_path / "ra"), random_seed=42)
    config_b = Config(models_dir=str(tmp_path / "b"), results_dir=str(tmp_path / "rb"), random_seed=42)

    result_a = run_training_pipeline(config_a)
    result_b = run_training_pipeline(config_b)

    assert result_a["summary"]["selected_model"] == result_b["summary"]["selected_model"]
    assert result_a["summary"]["test_metrics"]["MAE"] == result_b["summary"]["test_metrics"]["MAE"]
