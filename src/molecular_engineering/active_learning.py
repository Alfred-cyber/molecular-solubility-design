"""Simulated active-learning loop.

This module demonstrates the architecture of a closed-loop design cycle:

    train model -> search candidates -> select candidates
    -> get feedback -> add to training data -> retrain -> search again

The "feedback" here is a simulated measurement: the known dataset value for
that molecule plus random noise. This is a stand-in for what would, in
reality, come from a laboratory experiment. It is never labeled as real
experimental data anywhere in this module or in its output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .descriptors import build_feature_table
from .inverse_design import search_candidates
from .models import SolubilityPredictor


def simulate_experiment(true_value: float, noise_std: float = 0.3, random_state: np.random.Generator | None = None) -> float:
    """Return a noisy simulated measurement around a known dataset value.

    This is a simulation, not a laboratory result. noise_std controls how
    much synthetic measurement noise is added.
    """
    rng = random_state or np.random.default_rng()
    return float(true_value + rng.normal(0, noise_std))


def run_active_learning_loop(
    train_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    target_logS: float,
    model_name: str = "random_forest",
    rounds: int = 3,
    picks_per_round: int = 5,
    noise_std: float = 0.3,
    random_seed: int = 42,
) -> dict:
    """Run a small number of simulated active-learning rounds.

    train_df: labeled data to start from
    pool_df: unlabeled-in-spirit candidate pool with known 'smiles' and
        'target' columns; the 'target' is only used to generate the
        simulated measurement, mimicking what a real assay would return
    Returns a dict with the round-by-round validation-style MAE history
    and the final trained predictor, so the caller can inspect how
    performance changed as simulated data was added.
    """
    rng = np.random.default_rng(random_seed)
    current_train = train_df.copy()
    remaining_pool = pool_df.copy()
    history = []

    for round_idx in range(rounds):
        train_features = build_feature_table(current_train)
        predictor = SolubilityPredictor(model_name=model_name, random_seed=random_seed)
        predictor.fit(train_features, train_features["Target"])

        if remaining_pool.empty:
            break

        ranked = search_candidates(
            remaining_pool, predictor, target_logS, top_n=picks_per_round
        )
        picked_smiles = ranked["SMILES"].tolist()

        picked_rows = remaining_pool[remaining_pool["smiles"].isin(picked_smiles)]
        simulated_rows = []
        for _, row in picked_rows.iterrows():
            simulated_value = simulate_experiment(row["target"], noise_std=noise_std, random_state=rng)
            simulated_rows.append({"smiles": row["smiles"], "target": simulated_value})

        history.append(
            {
                "round": round_idx + 1,
                "train_size": len(current_train),
                "picked": picked_smiles,
                "mean_predicted_error": float(ranked["error"].mean()),
            }
        )

        current_train = pd.concat(
            [current_train, pd.DataFrame(simulated_rows)], ignore_index=True
        )
        remaining_pool = remaining_pool[~remaining_pool["smiles"].isin(picked_smiles)]

    final_features = build_feature_table(current_train)
    final_predictor = SolubilityPredictor(model_name=model_name, random_seed=random_seed)
    final_predictor.fit(final_features, final_features["Target"])

    return {
        "history": history,
        "final_predictor": final_predictor,
        "final_train_size": len(current_train),
    }
