"""Example: inverse design search for molecules near a target solubility.

Run with:
    python examples/inverse_design.py

Requires a trained model. Run examples/train_model.py first if
models/solubility_model.joblib does not exist yet.
"""

import pandas as pd

from molecular_engineering.config import load_config
from molecular_engineering.data_loader import load_delaney_splits
from molecular_engineering.inverse_design import search_candidates
from molecular_engineering.models import SolubilityPredictor
from molecular_engineering.optimization import optimize, Objective, Constraints
from molecular_engineering.visualization import plot_top_candidates


def main():
    config = load_config()
    predictor = SolubilityPredictor.load(config.model_path)

    splits = load_delaney_splits(
        dataset_name=config.dataset,
        smiles_column=config.smiles_column,
        target_column=config.target_column,
    )
    candidate_pool = pd.concat(
        [splits["train"], splits["validation"], splits["test"]], ignore_index=True
    )

    target_logS = config.target_logS
    print(f"Searching the dataset for molecules near target logS = {target_logS}")
    print("This ranks existing molecules; it does not generate new ones.\n")

    ranked = search_candidates(candidate_pool, predictor, target_logS, top_n=config.top_n)
    print(ranked.to_string(index=False))

    plot_top_candidates(ranked, target_logS, "results/top_candidates.png")

    print("\nSame search with constraints (max molecular weight 200, max LogP 3):")
    constrained = optimize(
        candidate_pool,
        predictor,
        Objective(kind="target", target_value=target_logS),
        constraints=Constraints(max_mw=200, max_logp=3),
        top_n=config.top_n,
    )
    print(constrained.to_string(index=False))


if __name__ == "__main__":
    main()
