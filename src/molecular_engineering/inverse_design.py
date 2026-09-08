"""Inverse molecular design: search a candidate pool for molecules whose
predicted property is closest to a desired target.

This first version searches real molecules that already exist in the
candidate pool (the Delaney dataset by default). It does not generate new
molecular structures. See MoleculeGenerator in this module for the
interface a future generative version would implement.

Workflow implemented here:

    target property -> search known chemical structures -> predict -> rank

A more advanced future workflow would be:

    target property -> generate new molecular structures -> validate
    -> predict -> optimize -> test

That distinction matters: ranking existing molecules is not the same as
discovering a genuinely novel one, and this module does not claim otherwise.
"""

from __future__ import annotations

import pandas as pd

from .descriptors import build_feature_table
from .models import SolubilityPredictor


class EmptyCandidatePoolError(Exception):
    """Raised when there are no valid candidates left to search."""


class MoleculeGenerator:
    """Interface for producing candidate molecules.

    The first implementation below simply returns molecules from a fixed
    dataset. It is deliberately not a generative model. A future
    implementation could plug in an evolutionary algorithm, a SELFIES-based
    sampler, a graph neural network, a molecular transformer, a diffusion
    model, or a reinforcement-learning policy, as long as it implements
    `generate(n)` and returns a list of SMILES strings.
    """

    def generate(self, n: int) -> list[str]:
        raise NotImplementedError


class DatasetMoleculeGenerator(MoleculeGenerator):
    """Returns SMILES sampled from a provided dataset, not newly invented ones."""

    def __init__(self, smiles_pool: list[str], random_seed: int = 42):
        self.smiles_pool = list(smiles_pool)
        self.random_seed = random_seed

    def generate(self, n: int) -> list[str]:
        import random

        rng = random.Random(self.random_seed)
        if n >= len(self.smiles_pool):
            return list(self.smiles_pool)
        return rng.sample(self.smiles_pool, n)


def search_candidates(
    candidate_df: pd.DataFrame,
    predictor: SolubilityPredictor,
    target_logS: float,
    top_n: int = 10,
    smiles_column: str = "smiles",
    target_column: str | None = "target",
) -> pd.DataFrame:
    """Rank candidate molecules by closeness of predicted logS to a target value.

    candidate_df must contain a SMILES column and, optionally, a measured
    target column for reference. Returns the top_n candidates sorted by
    ascending prediction error, with predicted and (if available) measured
    values, descriptors, and rank.
    """
    features = build_feature_table(
        candidate_df, smiles_column=smiles_column, target_column=target_column
    )
    if features.empty:
        raise EmptyCandidatePoolError("No valid candidate molecules after descriptor calculation")

    predicted = predictor.predict(features)
    features["predicted_logS"] = predicted
    features["error"] = (features["predicted_logS"] - target_logS).abs()

    if target_column is not None and "Target" in features.columns:
        features = features.rename(columns={"Target": "measured_logS"})
    else:
        features["measured_logS"] = None

    ranked = features.sort_values("error").reset_index(drop=True)
    ranked = ranked.head(top_n).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    columns = [
        "rank",
        "SMILES",
        "predicted_logS",
        "measured_logS",
        "error",
        "MW",
        "LogP",
        "TPSA",
    ]
    return ranked[columns]
