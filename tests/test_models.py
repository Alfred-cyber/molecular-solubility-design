import numpy as np
import pandas as pd
import pytest

from molecular_engineering.descriptors import build_feature_table
from molecular_engineering.models import SolubilityPredictor, ModelNotTrainedError


SMILES_POOL = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "CCCl", "CCCC", "OCC",
    "CC(C)O", "c1ccccc1O", "CCOCC", "CCCCCC", "CCCCO", "c1ccncc1", "CC(=O)OC",
    "CCCCCCCC", "c1ccc(O)cc1", "CC(C)(C)O", "CCCCCl", "OCCO",
]


def make_training_frame(seed=0):
    rng = np.random.default_rng(seed)
    fake_targets = rng.normal(-2.0, 1.0, size=len(SMILES_POOL))
    df = pd.DataFrame({"smiles": SMILES_POOL, "target": fake_targets})
    return build_feature_table(df)


def test_fit_predict_linear_regression():
    features = make_training_frame()
    model = SolubilityPredictor(model_name="linear_regression", random_seed=42)
    model.fit(features, features["Target"])
    preds = model.predict(features)
    assert len(preds) == len(features)
    assert np.isfinite(preds).all()


def test_predict_before_fit_raises():
    features = make_training_frame()
    model = SolubilityPredictor(model_name="random_forest")
    with pytest.raises(ModelNotTrainedError):
        model.predict(features)


def test_evaluate_before_fit_raises():
    features = make_training_frame()
    model = SolubilityPredictor(model_name="random_forest")
    with pytest.raises(ModelNotTrainedError):
        model.evaluate(features, features["Target"])


def test_evaluate_returns_expected_metric_keys():
    features = make_training_frame()
    model = SolubilityPredictor(model_name="gradient_boosting", random_seed=42)
    model.fit(features, features["Target"])
    metrics = model.evaluate(features, features["Target"])
    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        SolubilityPredictor(model_name="not_a_real_model")


def test_feature_importance_available_for_random_forest():
    features = make_training_frame()
    model = SolubilityPredictor(model_name="random_forest", random_seed=42)
    model.fit(features, features["Target"])
    importance = model.feature_importance()
    assert importance is not None
    assert len(importance) == len(model.feature_columns)


def test_save_and_load_round_trip(tmp_path):
    features = make_training_frame()
    model = SolubilityPredictor(model_name="random_forest", random_seed=42)
    model.fit(features, features["Target"])
    preds_before = model.predict(features)

    save_path = tmp_path / "model.joblib"
    model.save(save_path)

    reloaded = SolubilityPredictor.load(save_path)
    preds_after = reloaded.predict(features)

    np.testing.assert_allclose(preds_before, preds_after)


def test_save_before_fit_raises(tmp_path):
    model = SolubilityPredictor(model_name="random_forest")
    with pytest.raises(ModelNotTrainedError):
        model.save(tmp_path / "model.joblib")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SolubilityPredictor.load(tmp_path / "does_not_exist.joblib")


def test_same_seed_gives_reproducible_predictions():
    features = make_training_frame()
    model_a = SolubilityPredictor(model_name="random_forest", random_seed=7)
    model_a.fit(features, features["Target"])
    model_b = SolubilityPredictor(model_name="random_forest", random_seed=7)
    model_b.fit(features, features["Target"])
    np.testing.assert_allclose(model_a.predict(features), model_b.predict(features))
