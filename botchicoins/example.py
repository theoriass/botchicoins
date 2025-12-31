"""Exemplo executável do pipeline de análise preditiva para Bitcoin e Solana.

O fluxo segue etapas sugeridas em artigos científicos de *trend
forecasting* e *technical analysis*:

1. Construção de atributos técnicos e estatísticos.
2. Validação cruzada temporal com modelo de *ensemble* (Random Forest).
3. Conversão das previsões em sinais de alta exposição a risco e ganho.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd

from botchicoins.data_pipeline import PricePreprocessor
from botchicoins.model import CryptoTrendModel
from botchicoins.strategy import HighRiskHighRewardStrategy


def _simulate_price_series(symbol: str, days: int = 365) -> pd.DataFrame:
    """Gera série sintética de preços OHLCV via passeio aleatório."""

    rng = pd.date_range(end=datetime.utcnow(), periods=days, freq="D")
    drift = 0.0005 if symbol == "BTC" else 0.0008
    vol = 0.04 if symbol == "BTC" else 0.06
    returns = np.random.normal(loc=drift, scale=vol, size=len(rng))
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + np.random.normal(0, 0.005, len(close)))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.003, len(close))))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.003, len(close))))
    volume = np.random.randint(1_000, 10_000, size=len(close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=rng,
    )


def train_symbol(symbol: str) -> Dict[str, float]:
    """Treina modelo e retorna resumo de desempenho e parâmetros."""

    raw = _simulate_price_series(symbol)
    preprocessor = PricePreprocessor()
    enriched = preprocessor.add_features(raw)

    feature_cols = [
        "log_return",
        "direction",
        "volatility",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_width",
        "bb_position",
    ]

    train_df, test_df = preprocessor.train_test_split(enriched, test_size=0.25)
    X_train, y_train = preprocessor.feature_target_split(train_df, feature_cols)
    X_test, y_test = preprocessor.feature_target_split(test_df, feature_cols)

    model = CryptoTrendModel(feature_cols)
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    strategy = HighRiskHighRewardStrategy()
    signals = strategy.generate_signals(test_df, model.predict(X_test))
    metrics.update(strategy.performance_summary(signals))
    return metrics


def run_demo() -> None:
    btc_metrics = train_symbol("BTC")
    sol_metrics = train_symbol("SOL")

    print("Resumo BTC:")
    for k, v in btc_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nResumo SOL:")
    for k, v in sol_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    run_demo()
