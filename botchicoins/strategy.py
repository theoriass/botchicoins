"""Estratégias de decisão baseadas em risco elevado e alto retorno."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class HighRiskHighRewardStrategy:
    """Transforma previsões de retorno em sinais de compra e venda.

    A estratégia favorece operações com maior assimetria positiva,
    procurando trades agressivos: compra quando a previsão supera o
    quantil alto dos retornos históricos recentes e momentum é
    favorável; vende/short quando a previsão é fortemente negativa.
    """

    long_quantile: float = 0.7
    short_quantile: float = 0.3
    min_volatility: float = 1e-4

    def generate_signals(
        self, frame: pd.DataFrame, predictions: np.ndarray
    ) -> pd.DataFrame:
        """Anexa sinais ``long``/``short`` ao *DataFrame* baseado nos limiares."""

        if len(frame) != len(predictions):
            raise ValueError("Número de previsões deve igualar número de linhas de frame.")

        result = frame.copy()
        result["predicted_return"] = predictions

        long_threshold = frame["log_return"].quantile(self.long_quantile)
        short_threshold = frame["log_return"].quantile(self.short_quantile)

        result["signal"] = "flat"
        result.loc[
            (result["predicted_return"] > long_threshold)
            & (result["macd_hist"] > 0)
            & (result["rsi"] > 55)
            & (result["volatility"] > self.min_volatility),
            "signal",
        ] = "long"
        result.loc[
            (result["predicted_return"] < short_threshold)
            & (result["macd_hist"] < 0)
            & (result["rsi"] < 45)
            & (result["volatility"] > self.min_volatility),
            "signal",
        ] = "short"

        return result

    @staticmethod
    def performance_summary(trades: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas básicas de rentabilidade acumulada."""

        pnl = trades["signal"].map({"long": 1, "short": -1, "flat": 0}) * trades[
            "log_return"
        ]
        cumulative_return = float(np.exp(pnl.sum()) - 1)
        hit_ratio = float((pnl > 0).mean())
        return {"cumulative_return": cumulative_return, "hit_ratio": hit_ratio}
