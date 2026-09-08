import numpy as np
import pandas as pd
import pytest

from molecular_engineering.descriptors import build_feature_table
from molecular_engineering.models import SolubilityPredictor
from molecular_engineering.optimization import optimize, Objective, Constraints

SMILES_POOL = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "CCCl", "CCCC", "OCC",
    "CC(C)O", "c1ccccc1O", "CCOCC", "CCCCCC", "CCCCO", "c1ccncc1", "CC(=O)OC",
]


def trained_predictor():
    rng = np.random.default_rng(0)
    targets = rng.normal(-2.0, 1.0, size=len(SMILES_POOL))
    df = pd.DataFrame({"smiles": SMILES_POOL, "target": targets})
    features = build_feature_table(df)
    model = SolubilityPredictor(model_name="random_forest", random_seed=42)
    model.fit(features, features["Target"])
    return model, df


def test_optimize_target_objective_ranks_by_absolute_error():
    model, df = trained_predictor()
    result = optimize(df, model, Objective(kind="target", target_value=-2.0), top_n=5)
    scores = result["score"].tolist()
    assert scores == sorted(scores)


def test_optimize_minimize_objective_orders_ascending():
    model, df = trained_predictor()
    result = optimize(df, model, Objective(kind="minimize"), top_n=5)
    preds = result["predicted_logS"].tolist()
    assert preds == sorted(preds)


def test_optimize_maximize_objective_orders_descending():
    model, df = trained_predictor()
    result = optimize(df, model, Objective(kind="maximize"), top_n=5)
    preds = result["predicted_logS"].tolist()
    assert preds == sorted(preds, reverse=True)


def test_optimize_respects_max_mw_constraint():
    model, df = trained_predictor()
    constraints = Constraints(max_mw=50)
    result = optimize(df, model, Objective(kind="minimize"), constraints=constraints, top_n=20)
    assert (result["MW"] <= 50).all()


def test_optimize_all_candidates_eliminated_raises():
    model, df = trained_predictor()
    constraints = Constraints(max_mw=1)
    with pytest.raises(ValueError):
        optimize(df, model, Objective(kind="minimize"), constraints=constraints)


def test_objective_target_requires_target_value():
    with pytest.raises(ValueError):
        Objective(kind="target")


def test_objective_unsupported_kind_raises():
    with pytest.raises(ValueError):
        Objective(kind="not_a_real_kind")
