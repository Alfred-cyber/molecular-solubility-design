"""Example: train the solubility model end to end and print a summary.

Run with:
    python examples/train_model.py
"""

from molecular_engineering.config import load_config
from molecular_engineering.pipeline import run_training_pipeline
from molecular_engineering.visualization import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_target_distribution,
    plot_feature_vs_target,
)


def main():
    config = load_config()
    result = run_training_pipeline(config)

    print("Model comparison (validation performance):")
    print(result["comparison"].to_string(index=False))

    summary = result["summary"]
    print(f"\nSelected model: {summary['selected_model']}")
    print(f"Test set metrics: {summary['test_metrics']}")

    test_features = result["features"]["test"]
    predictions = result["predictor"].predict(test_features)

    plot_actual_vs_predicted(test_features["Target"], predictions, "results/actual_vs_predicted.png")
    plot_residuals(test_features["Target"], predictions, "results/residuals.png")

    train_features = result["features"]["train"]
    plot_target_distribution(train_features["Target"], "results/target_distribution.png")
    plot_feature_vs_target(train_features["MW"], train_features["Target"], "MW", "results/mw_vs_logS.png")
    plot_feature_vs_target(train_features["LogP"], train_features["Target"], "LogP", "results/logp_vs_logS.png")

    print("\nPlots written to the results/ directory.")


if __name__ == "__main__":
    main()
