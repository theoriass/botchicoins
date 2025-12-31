"""Modelagem preditiva de tendências para Bitcoin e Solana.

O modelo combina atributos técnicos clássicos com um regressor de
floresta aleatória, seguindo evidências da literatura de *forecasting*
que recomendam ensembles para séries financeiras devido à robustez a
não-linearidades e *overfitting* em janelas curtas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CryptoTrendModel:
    """Modelo de previsão de retorno logarítmico de curto prazo."""

    feature_columns: Iterable[str]
    n_estimators: int = 400
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2
    random_state: int = 42
    n_splits: int = 5

    pipeline: Pipeline = field(init=False)

    def __post_init__(self) -> None:
        scaler = ColumnTransformer([
            ("num", StandardScaler(), list(self.feature_columns)),
        ])
        regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.pipeline = Pipeline([
            ("scaler", scaler),
            ("model", regressor),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Treina o modelo com validação cruzada temporal para evitar *leakage*."""

        self._validate_inputs(X, y)
        cv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = cross_val_score(
            self.pipeline,
            X,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        # Ajusta modelo final com todos os dados
        self.pipeline.fit(X, y)
        self.cv_rmse_ = -scores.mean()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna o retorno logarítmico previsto para o próximo período."""

        return self.pipeline.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calcula métricas clássicas de previsão."""

        preds = self.predict(X_test)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds)),
            "cv_rmse": float(getattr(self, "cv_rmse_", np.nan)),
        }

    @staticmethod
    def _validate_inputs(X: pd.DataFrame, y: pd.Series) -> None:
        if len(X) != len(y):
            raise ValueError("X e y devem ter o mesmo número de linhas.")
        if X.isna().any().any() or y.isna().any():
            raise ValueError("Remova valores nulos antes do treinamento.")
