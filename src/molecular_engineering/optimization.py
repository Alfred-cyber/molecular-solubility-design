"""Generic scoring and constraint logic for inverse design.

This is intentionally simple: score each candidate against an objective,
drop anything that violates a hard constraint, and return the best-scoring
survivors. It is written to be extendable to properties beyond solubility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .descriptors import build_feature_table
from .models import SolubilityPredictor
from .molecules import parse_smiles


@dataclass
class Objective:
    """Describes what to optimize for.

    kind: "target", "minimize", or "maximize"
    target_value: required when kind == "target"
    tolerance: optional band around target_value; candidates outside it are
        dropped rather than merely penalized
    """

    kind: str
    target_value: float | None = None
    tolerance: float | None = None

    def __post_init__(self):
        if self.kind not in ("target", "minimize", "maximize"):
            raise ValueError(f"Unsupported objective kind: {self.kind}")
        if self.kind == "target" and self.target_value is None:
            raise ValueError("target_value is required for a 'target' objective")


@dataclass
class Constraints:
    max_mw: float | None = None
    max_logp: float | None = None
    min_tpsa: float | None = None
    max_rings: int | None = None
    allowed_elements: set[str] | None = None


def _elements_in_smiles(smiles: str) -> set[str]:
    mol = parse_smiles(smiles)
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def _violates_constraints(row: pd.Series, smiles: str, constraints: Constraints) -> bool:
    if constraints.max_mw is not None and row["MW"] > constraints.max_mw:
        return True
    if constraints.max_logp is not None and row["LogP"] > constraints.max_logp:
        return True
    if constraints.min_tpsa is not None and row["TPSA"] < constraints.min_tpsa:
        return True
    if constraints.max_rings is not None and row["RingCount"] > constraints.max_rings:
        return True
    if constraints.allowed_elements is not None:
        elements = _elements_in_smiles(smiles)
        if not elements.issubset(constraints.allowed_elements):
            return True
    return False


def score_candidates(
    predicted_values: pd.Series, objective: Objective
) -> pd.Series:
    """Lower score is better in all three objective kinds."""
    if objective.kind == "target":
        return (predicted_values - objective.target_value).abs()
    if objective.kind == "minimize":
        return predicted_values
    if objective.kind == "maximize":
        return -predicted_values
    raise ValueError(f"Unsupported objective kind: {objective.kind}")


def optimize(
    candidate_df: pd.DataFrame,
    predictor: SolubilityPredictor,
    objective: Objective,
    constraints: Constraints | None = None,
    top_n: int = 10,
    smiles_column: str = "smiles",
    target_column: str | None = "target",
) -> pd.DataFrame:
    """Run the full search -> predict -> constrain -> score -> rank pipeline."""
    constraints = constraints or Constraints()

    features = build_feature_table(
        candidate_df, smiles_column=smiles_column, target_column=target_column
    )
    if features.empty:
        raise ValueError("No valid candidates after descriptor calculation")

    features["predicted_logS"] = predictor.predict(features)

    keep_mask = ~features.apply(
        lambda row: _violates_constraints(row, row["SMILES"], constraints), axis=1
    )
    survivors = features[keep_mask].copy()
    if survivors.empty:
        raise ValueError("All candidates were eliminated by the given constraints")

    if objective.kind == "target" and objective.tolerance is not None:
        within = (survivors["predicted_logS"] - objective.target_value).abs() <= objective.tolerance
        survivors = survivors[within].copy()
        if survivors.empty:
            raise ValueError("No candidates fall within the target tolerance")

    survivors["score"] = score_candidates(survivors["predicted_logS"], objective)
    ranked = survivors.sort_values("score").reset_index(drop=True).head(top_n)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
