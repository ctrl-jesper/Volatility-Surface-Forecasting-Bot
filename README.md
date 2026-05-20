# Volatility Surface Forecasting Bot

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Domain](https://img.shields.io/badge/Domain-Derivatives%20%7C%20ML-FF6F00?style=flat)

> Predict implied volatility surfaces across the full strike × maturity grid using Random Forest and Gradient Boosting regression.

---

## Overview

Implied volatility is not constant across strikes and maturities — it forms a curved surface driven by supply/demand, skew, and term structure effects. This model learns to predict that surface from engineered features including moneyness, log-moneyness, √T, and a historical vol proxy.

The fitted surface is rendered as an interactive 3-D plot alongside actual vs predicted scatter and feature importances.

---

## Features

- **Two ML backends:** Random Forest (default) and Gradient Boosting — switchable via `--model gb`
- Realistic vol smile simulation with negative skew and term structure inversion
- Five engineered features: moneyness, log-moneyness, maturity, √maturity, historical vol
- Evaluation metrics: MSE and R² on held-out test set
- Three-panel visualisation: 3-D surface · actual vs predicted · feature importance
- Accepts real options data CSV or runs on built-in synthetic data

---

## Installation

```bash
git clone https://github.com/ctrl-jesper/Volatility-Surface-Forecasting-Bot.git
cd Volatility-Surface-Forecasting-Bot
pip install -r requirements.txt
```

**Requirements:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`

---

## Usage

```bash
# Synthetic data, Random Forest (default)
python src/main.py

# Gradient Boosting backend
python src/main.py --model gb

# Real options data (CSV must have columns: strike, maturity, implied_volatility, spot)
python src/main.py --data options.csv --model rf
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `rf` | Model backend: `rf` (Random Forest) or `gb` (Gradient Boosting) |
| `--data` | `None` | Path to CSV with real options data |
| `--test-size` | `0.2` | Fraction of data held out for evaluation |
| `--n-estimators` | `100` | Number of trees / boosting rounds |

---

## How It Works

### Feature Engineering

Raw market data is transformed into five predictive features:

| Feature | Formula | Rationale |
|---|---|---|
| Moneyness | `S / K` | Captures smile asymmetry |
| Log-moneyness | `ln(S / K)` | Symmetric, scale-invariant skew measure |
| Maturity | `T` | Linear term structure component |
| √Maturity | `√T` | Non-linear term structure component |
| Historical vol | Rolling std of log returns | Regime-conditioning signal |

### Model Training

1. Simulate (or load) a vol surface with realistic smile and term structure
2. Engineer features for each `(strike, maturity)` observation
3. Train the chosen regressor on 80% of observations
4. Evaluate MSE and R² on the held-out 20%
5. Reconstruct the full predicted surface on a dense grid for visualisation

---

## Project Structure

```
src/
├── main.py          # Entry point with CLI argument parsing
├── data.py          # Synthetic data simulation + CSV loader
├── features.py      # Feature engineering pipeline
├── model.py         # VolSurfaceModel class (RF + GB backends)
├── surface.py       # Dense grid construction for surface rendering
└── visualization.py # 3-panel dark-themed figure
requirements.txt
README.md
```

---

## Output

Three-panel figure:

| Panel | Content |
|---|---|
| 3-D surface | Predicted implied vol across the full (strike × maturity) grid |
| Actual vs predicted | Scatter plot with R² annotation |
| Feature importance | Bar chart of relative predictor weights |

---

## References

- [Implied volatility — Wikipedia](https://en.wikipedia.org/wiki/Implied_volatility)
- [Volatility surface — Wikipedia](https://en.wikipedia.org/wiki/Volatility_surface)
- [Random forest — Wikipedia](https://en.wikipedia.org/wiki/Random_forest)
- [Gradient boosting — Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting)

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. Ensure new code includes docstrings and follows the existing module structure.

---

## License

MIT — see [LICENSE](LICENSE) for details.

**Author:** [Jesper Mathias Nielsen](https://github.com/ctrl-jesper)
