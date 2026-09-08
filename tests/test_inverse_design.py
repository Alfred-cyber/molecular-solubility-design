import numpy as np
import pandas as pd
import pytest

from molecular_engineering.descriptors import build_feature_table
from molecular_engineering.models import SolubilityPredictor
from molecular_engineering.inverse_design import (
    search_candidates,
    EmptyCandidatePoolError,
    DatasetMoleculeGenerator,
)

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


def test_search_candidates_returns_ranked_results():
    model, df = trained_predictor()
    ranked = search_candidates(df, model, target_logS=-2.0, top_n=5)
    assert len(ranked) == 5
    assert list(ranked["rank"]) == [1, 2, 3, 4, 5]
    errors = ranked["error"].tolist()
    assert errors == sorted(errors)


def test_search_candidates_error_matches_definition():
    model, df = trained_predictor()
    ranked = search_candidates(df, model, target_logS=-1.5, top_n=3)
    for _, row in ranked.iterrows():
        expected_error = abs(row["predicted_logS"] - (-1.5))
        assert row["error"] == pytest.approx(expected_error)


def test_search_candidates_empty_pool_raises():
    model, _ = trained_predictor()
    empty_df = pd.DataFrame({"smiles": ["not_valid_smiles!!!"], "target": [0.0]})
    with pytest.raises(EmptyCandidatePoolError):
        search_candidates(empty_df, model, target_logS=-2.0)


def test_dataset_molecule_generator_returns_pool_subset():
    generator = DatasetMoleculeGenerator(SMILES_POOL, random_seed=1)
    generated = generator.generate(5)
    assert len(generated) == 5
    assert all(s in SMILES_POOL for s in generated)


def test_dataset_molecule_generator_reproducible_with_seed():
    gen_a = DatasetMoleculeGenerator(SMILES_POOL, random_seed=3)
    gen_b = DatasetMoleculeGenerator(SMILES_POOL, random_seed=3)
    assert gen_a.generate(5) == gen_b.generate(5)
