"""End-to-end training pipeline: load data, build features, compare
baseline models, select the best one on validation performance, and
evaluate once on the held-out test set.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .data_loader import load_delaney_splits
from .descriptors import build_feature_table, DESCRIPTOR_COLUMNS
from .evaluation import compare_models
from .models import SolubilityPredictor, BASELINE_MODELS


def run_training_pipeline(config: Config) -> dict:
    """Run the full training workflow and return a results summary.

    Also saves the selected model to config.model_path and writes a
    results record to config.results_dir.
    """
    splits = load_delaney_splits(
        dataset_name=config.dataset,
        smiles_column=config.smiles_column,
        target_column=config.target_column,
    )

    train_features = build_feature_table(splits["train"])
    val_features = build_feature_table(splits["validation"])
    test_features = build_feature_table(splits["test"])

    candidates = {
        name: SolubilityPredictor(model_name=name, random_seed=config.random_seed)
        for name in BASELINE_MODELS
    }
    comparison = compare_models(candidates, train_features, val_features, DESCRIPTOR_COLUMNS)

    best_model_name = comparison.iloc[0]["model"]
    best_predictor = SolubilityPredictor(model_name=best_model_name, random_seed=config.random_seed)
    best_predictor.fit(train_features, train_features["Target"])

    test_metrics = best_predictor.evaluate(test_features, test_features["Target"])

    best_predictor.save(config.model_path)

    summary = {
        "dataset": config.dataset,
        "random_seed": config.random_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_size": len(train_features),
        "validation_size": len(val_features),
        "test_size": len(test_features),
        "model_comparison": comparison.to_dict(orient="records"),
        "selected_model": best_model_name,
        "test_metrics": test_metrics,
        "model_path": str(config.model_path),
    }

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": summary,
        "predictor": best_predictor,
        "splits": splits,
        "features": {
            "train": train_features,
            "validation": val_features,
            "test": test_features,
        },
        "comparison": comparison,
    }
