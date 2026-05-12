# Volatility Surface Forecasting

**Author: [Jesper Mathias Nielsen](https://github.com/jespermathiasnielsen)**

Predicts implied volatility surfaces across the full (strike × maturity) grid using machine learning regression — Random Forest and Gradient Boosting backends.

## What it does

Implied volatility is not constant across strikes and maturities — it forms a curved surface driven by supply/demand, skew, and term structure. This model learns to predict that surface from engineered features including moneyness, log-moneyness, √T, and a historical vol proxy. The fitted surface is rendered as an interactive 3-D plot alongside actual vs predicted scatter and feature importances.

## Features

- **Two model backends**: Random Forest (default) and Gradient Boosting — switch with `--model gb`
- Realistic vol smile simulation: negative skew + term structure inversion
- Five engineered features: moneyness, log-moneyness, maturity, √maturity, historical vol
- Evaluation: MSE and R² on held-out test set
- Three-panel visualisation: 3-D surface, actual vs predicted, feature importance
- Works with real options data CSV or built-in synthetic data

## Setup

```bash
git clone https://github.com/jespermathiasnielsen/Volatility-Surface-Forecasting-Bot.git
cd Volatility-Surface-Forecasting-Bot
pip install -r requirements.txt

# Run with synthetic data (Random Forest)
python src/main.py

# Run with Gradient Boosting
python src/main.py --model gb

# Run with real options data (columns: strike, maturity, implied_volatility, spot)
python src/main.py --data options.csv
```

## Project structure

```
src/
├── main.py          # Entry point with CLI arguments
├── data.py          # Simulation + CSV loader
├── features.py      # Feature engineering (moneyness, log-moneyness, etc.)
├── model.py         # VolSurfaceModel class (RF + GB)
├── surface.py       # Grid creation for surface rendering
└── visualization.py # 3-panel dark-themed figure
```

## Concepts

- [Implied volatility](https://en.wikipedia.org/wiki/Implied_volatility)
- [Volatility surface](https://en.wikipedia.org/wiki/Volatility_surface)
- [Random forest](https://en.wikipedia.org/wiki/Random_forest)
- [Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)

## License

MIT
