# Bot de análise preditiva para Bitcoin e Solana

Este repositório contém um pipeline de **análise estatística e técnica**
para capturar tendências de preços em Bitcoin e Solana. O foco é
combinar métodos usados em artigos científicos de *forecasting* de ativos
voláteis: engenharia de atributos robusta, validação cruzada temporal e
modelos de ensemble para prever retornos logarítmicos.

## Arquitetura
- `botchicoins/data_pipeline.py`: pré-processamento de OHLCV, cálculo de
  retornos logarítmicos, volatilidade realizada, RSI, MACD e Bandas de
  Bollinger.
- `botchicoins/model.py`: modelo `RandomForestRegressor` encapsulado em
  *pipeline* com *scaling* e validação cruzada em séries temporais.
- `botchicoins/strategy.py`: regra de decisão "alto risco e alto ganho",
  usando quantis de retorno para definir compras (`long`) e vendas a
  descoberto (`short`).
- `botchicoins/example.py`: demonstração completa com séries sintéticas
  para BTC e SOL.

## Instalação
```bash
pip install -r requirements.txt
```

## Uso rápido
Execute a demonstração com dados sintéticos:

```bash
python -m botchicoins.example
```

Para usar dados reais, forneça um `DataFrame` com colunas
`['open', 'high', 'low', 'close', 'volume']` e índice `DatetimeIndex`
para `PricePreprocessor.add_features`, treine o modelo e gere sinais com
`HighRiskHighRewardStrategy`.
