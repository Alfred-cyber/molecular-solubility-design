"""Command-line interface for the molecular engineering project.

Examples:
    python -m molecular_engineering.cli train
    python -m molecular_engineering.cli predict "CCO"
    python -m molecular_engineering.cli search --target -2.0 --top 10
    python -m molecular_engineering.cli evaluate
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import load_config
from .data_loader import load_delaney_splits
from .descriptors import compute_descriptors, build_feature_table
from .inverse_design import search_candidates
from .models import SolubilityPredictor, ModelNotTrainedError
from .molecules import InvalidSMILESError
from .pipeline import run_training_pipeline


def cmd_train(args):
    config = load_config(args.config)
    result = run_training_pipeline(config)
    summary = result["summary"]

    print("\nModel comparison (sorted by validation MAE):")
    print(result["comparison"].to_string(index=False))

    print(f"\nSelected model: {summary['selected_model']}")
    print("Test set performance:")
    for name, value in summary["test_metrics"].items():
        print(f"  {name}: {value:.4f}")

    importance = result["predictor"].feature_importance()
    if importance is not None:
        print("\nModel-level feature importance (does not imply causation):")
        print(importance.to_string())

    print(f"\nModel saved to: {summary['model_path']}")


def cmd_predict(args):
    config = load_config(args.config)
    try:
        desc = compute_descriptors(args.smiles)
    except InvalidSMILESError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        predictor = SolubilityPredictor.load(config.model_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}. Run 'python -m molecular_engineering.cli train' first.", file=sys.stderr)
        sys.exit(1)

    row = pd.DataFrame([desc])
    prediction = predictor.predict(row)[0]

    print(f"SMILES: {args.smiles}")
    print(f"Predicted logS (model output, not a measurement): {prediction:.3f}")


def cmd_search(args):
    config = load_config(args.config)
    try:
        predictor = SolubilityPredictor.load(config.model_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}. Run 'python -m molecular_engineering.cli train' first.", file=sys.stderr)
        sys.exit(1)

    splits = load_delaney_splits(
        dataset_name=config.dataset,
        smiles_column=config.smiles_column,
        target_column=config.target_column,
    )
    candidate_pool = pd.concat([splits["train"], splits["validation"], splits["test"]], ignore_index=True)

    ranked = search_candidates(candidate_pool, predictor, args.target, top_n=args.top)
    print(f"Top {args.top} candidates for target logS = {args.target}:")
    print(
        "This searches known molecules from the dataset and ranks them by "
        "the trained model's prediction. It does not generate new molecules."
    )
    print(ranked.to_string(index=False))


def cmd_evaluate(args):
    config = load_config(args.config)
    try:
        predictor = SolubilityPredictor.load(config.model_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}. Run 'python -m molecular_engineering.cli train' first.", file=sys.stderr)
        sys.exit(1)

    splits = load_delaney_splits(
        dataset_name=config.dataset,
        smiles_column=config.smiles_column,
        target_column=config.target_column,
    )
    test_features = build_feature_table(splits["test"])
    metrics = predictor.evaluate(test_features, test_features["Target"])

    print("Test set evaluation:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molecular_engineering", description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a JSON or YAML config file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save the solubility model")
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Predict logS for a SMILES string")
    predict_parser.add_argument("smiles", type=str)
    predict_parser.set_defaults(func=cmd_predict)

    search_parser = subparsers.add_parser("search", help="Search for candidates near a target logS")
    search_parser.add_argument("--target", type=float, required=True)
    search_parser.add_argument("--top", type=int, default=10)
    search_parser.set_defaults(func=cmd_search)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the saved model on the test set")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
