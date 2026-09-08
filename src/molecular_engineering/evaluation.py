"""Regression metrics and baseline model comparison utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def compare_models(models: dict, train_df, val_df, feature_columns, target_column="Target") -> pd.DataFrame:
    """Fit each candidate model on train and score it on train and validation.

    `models` maps a name to an already-constructed SolubilityPredictor.
    Returns a dataframe with one row per model, sorted by validation MAE.
    Model selection should use the validation columns, not the train columns.
    """
    rows = []
    for name, predictor in models.items():
        predictor.fit(train_df, train_df[target_column])
        train_metrics = predictor.evaluate(train_df, train_df[target_column])
        val_metrics = predictor.evaluate(val_df, val_df[target_column])
        rows.append(
            {
                "model": name,
                "train_MAE": train_metrics["MAE"],
                "train_RMSE": train_metrics["RMSE"],
                "train_R2": train_metrics["R2"],
                "val_MAE": val_metrics["MAE"],
                "val_RMSE": val_metrics["RMSE"],
                "val_R2": val_metrics["R2"],
            }
        )
    result = pd.DataFrame(rows).sort_values("val_MAE").reset_index(drop=True)
    return result
