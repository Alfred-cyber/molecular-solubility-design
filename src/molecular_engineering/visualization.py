"""Plotting helpers for model evaluation and inverse-design results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, out_path: str | Path | None):
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(y_true, y_pred, out_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Measured logS")
    ax.set_ylabel("Predicted logS")
    ax.set_title("Actual vs predicted solubility")
    return _save(fig, out_path)


def plot_residuals(y_true, y_pred, out_path: str | Path | None = None):
    residuals = pd.Series(y_pred) - pd.Series(y_true).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(range(len(residuals)), residuals, alpha=0.5, s=15)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Residual (predicted - measured)")
    ax.set_title("Prediction residuals")
    return _save(fig, out_path)


def plot_target_distribution(target: pd.Series, out_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(target, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Measured logS")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of measured solubility")
    return _save(fig, out_path)


def plot_feature_vs_target(feature_values: pd.Series, target: pd.Series, feature_name: str,
                            out_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(feature_values, target, alpha=0.5, s=15)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Measured logS")
    ax.set_title(f"{feature_name} vs measured logS")
    return _save(fig, out_path)


def plot_top_candidates(candidates: pd.DataFrame, target_logS: float, out_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(candidates))))
    labels = candidates["SMILES"]
    ax.barh(labels, candidates["predicted_logS"], color="steelblue")
    ax.axvline(target_logS, color="r", linestyle="--", label="target logS")
    ax.set_xlabel("Predicted logS")
    ax.set_title("Top inverse-design candidates")
    ax.invert_yaxis()
    ax.legend()
    return _save(fig, out_path)
