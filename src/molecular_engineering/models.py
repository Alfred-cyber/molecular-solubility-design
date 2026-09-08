"""Solubility prediction models.

Provides a small set of baseline regressors and a SolubilityPredictor
wrapper class that handles fitting, prediction, evaluation, and
persistence through a scikit-learn pipeline.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .descriptors import DESCRIPTOR_COLUMNS
from .evaluation import regression_metrics

BASELINE_MODELS = {
    "linear_regression": lambda seed: LinearRegression(),
    "random_forest": lambda seed: RandomForestRegressor(
        n_estimators=300, random_state=seed, n_jobs=-1
    ),
    "gradient_boosting": lambda seed: HistGradientBoostingRegressor(random_state=seed),
}


class ModelNotTrainedError(Exception):
    """Raised when predict/evaluate/save is called before fit."""


class SolubilityPredictor:
    """Predicts logS from molecular descriptors using a chosen regressor.

    The pipeline always standardizes the descriptor features before the
    linear model. Tree-based models do not require scaling but it does no
    harm and keeps the pipeline uniform.
    """

    def __init__(self, model_name: str = "random_forest", random_seed: int = 42):
        if model_name not in BASELINE_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from {list(BASELINE_MODELS)}"
            )
        self.model_name = model_name
        self.random_seed = random_seed
        self.feature_columns = list(DESCRIPTOR_COLUMNS)
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", BASELINE_MODELS[model_name](random_seed)),
            ]
        )
        self._is_trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SolubilityPredictor":
        self.pipeline.fit(X[self.feature_columns], y)
        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_trained:
            raise ModelNotTrainedError("Call fit() before predict()")
        return self.pipeline.predict(X[self.feature_columns])

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        if not self._is_trained:
            raise ModelNotTrainedError("Call fit() before evaluate()")
        preds = self.predict(X)
        return regression_metrics(y, preds)

    def feature_importance(self) -> pd.Series | None:
        """Return model-level feature importance if the underlying model supports it.

        This reflects how much the model relies on each descriptor for its
        predictions. It does not prove that a descriptor causes solubility
        to change.
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Call fit() before feature_importance()")
        regressor = self.pipeline.named_steps["regressor"]
        if hasattr(regressor, "feature_importances_"):
            values = regressor.feature_importances_
        elif hasattr(regressor, "coef_"):
            values = np.abs(regressor.coef_)
        else:
            return None
        return pd.Series(values, index=self.feature_columns).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        if not self._is_trained:
            raise ModelNotTrainedError("Call fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "model_name": self.model_name,
                "random_seed": self.random_seed,
                "feature_columns": self.feature_columns,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SolubilityPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved model found at: {path}")
        payload = joblib.load(path)
        predictor = cls(model_name=payload["model_name"], random_seed=payload["random_seed"])
        predictor.pipeline = payload["pipeline"]
        predictor.feature_columns = payload["feature_columns"]
        predictor._is_trained = True
        return predictor
