"""Rotinas de pré-processamento e engenharia de atributos para séries de preços.

Inspirado em literatura de finanças quantitativas, o módulo implementa
indicadores técnicos consagrados (RSI, MACD, Bandas de Bollinger) e
métricas estatísticas como retornos logarítmicos e volatilidade
realizada. As funções retornam *DataFrames* prontos para uso em
modelagem preditiva.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


@dataclass
class PricePreprocessor:
    """Gera atributos previsores a partir de OHLCV.

    Espera um ``DataFrame`` com colunas ``['open', 'high', 'low', 'close', 'volume']``
    e índice ``DatetimeIndex``. Todos os indicadores são calculados com
    janelas parametrizáveis para facilitar experimentação.
    """

    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volatility_window: int = 20
    bollinger_window: int = 20
    bollinger_std: float = 2.0

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        missing = set(["open", "high", "low", "close", "volume"]) - set(frame.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("O índice deve ser um DatetimeIndex para operações temporais.")

    def add_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Retorna um novo *DataFrame* com atributos técnicos e estatísticos.

        Os atributos incluem:
        - Retorno logarítmico e direção do retorno (sinal)
        - Volatilidade realizada (desvio-padrão móvel dos retornos)
        - Indicadores de momentum (RSI, MACD e sua linha de sinal)
        - Bandas de Bollinger e distância do preço às bandas
        """

        self._validate_columns(frame)
        df = frame.copy().sort_index()

        df["log_return"] = np.log(df["close"]).diff()
        df["direction"] = np.sign(df["log_return"])

        df["volatility"] = df["log_return"].rolling(self.volatility_window).std()

        rsi = RSIIndicator(close=df["close"], window=self.rsi_window)
        df["rsi"] = rsi.rsi()

        macd = MACD(
            close=df["close"],
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal,
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        bb = BollingerBands(
            close=df["close"],
            window=self.bollinger_window,
            window_dev=self.bollinger_std,
        )
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]
        df["bb_position"] = (df["close"] - df["bb_low"]) / (
            (df["bb_high"] - df["bb_low"]).replace(0, np.nan)
        )

        # Previsão mira o retorno seguinte
        df["target"] = df["log_return"].shift(-1)

        df = df.dropna()
        return df

    @staticmethod
    def train_test_split(
        frame: pd.DataFrame, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Divide a série cronologicamente preservando a ordem temporal."""

        if not 0 < test_size < 1:
            raise ValueError("test_size deve estar entre 0 e 1.")
        split_idx = int(len(frame) * (1 - test_size))
        train, test = frame.iloc[:split_idx], frame.iloc[split_idx:]
        if len(train) == 0 or len(test) == 0:
            raise ValueError("Particionamento resultou em divisão vazia; aumente os dados.")
        return train, test

    @staticmethod
    def feature_target_split(
        frame: pd.DataFrame, feature_columns: Iterable[str]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separa preditores e alvo (retorno futuro)."""

        X = frame.loc[:, list(feature_columns)]
        y = frame["target"]
        return X, y
