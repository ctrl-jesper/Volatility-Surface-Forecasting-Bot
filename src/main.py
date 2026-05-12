"""
Volatility Surface Forecasting — entry point.

Trains a Random Forest (or Gradient Boosting) regressor to predict implied
volatility across a (strike, maturity) grid and renders the resulting surface.

Author: Jesper Mathias Nielsen
Usage:
    python src/main.py                # synthetic data, Random Forest
    python src/main.py --model gb     # Gradient Boosting
    python src/main.py --data opts.csv
"""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from data import load_data
from features import add_features, FEATURE_COLS
from model import VolSurfaceModel
from surface import create_surface_grid
from visualization import plot_all_vertical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Volatility Surface Forecasting")
    parser.add_argument("--data",  type=str, default=None, help="Path to options CSV file")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "gb"],
                        help="Model backend: 'rf' (Random Forest) or 'gb' (Gradient Boosting)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading {'real' if args.data else 'synthetic'} options data …")
    df = load_data(args.data)
    df = add_features(df)

    target  = "implied_volatility"
    X = df[FEATURE_COLS].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Training {args.model.upper()} model …")
    model = VolSurfaceModel(model_type=args.model)
    model.train(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    mse, r2, y_pred = model.evaluate(X_test, y_test)
    print(f"\nHold-out metrics  —  MSE: {mse:.6f}  |  R²: {r2:.4f}")

    # ── Build prediction grid ─────────────────────────────────────────────────
    spot = float(df["spot"].iloc[0])
    grid_df, grid_strike, grid_maturity = create_surface_grid(
        strike_range   = (df["strike"].min(),   df["strike"].max()),
        maturity_range = (df["maturity"].min(), df["maturity"].max()),
        spot=spot,
    )
    grid_df = add_features(grid_df)
    Z_pred  = model.predict(grid_df[FEATURE_COLS].values).reshape(grid_strike.shape)

    # ── Visualise ─────────────────────────────────────────────────────────────
    plot_all_vertical(
        X=grid_strike,
        Y=grid_maturity,
        Z=Z_pred,
        y_true=y_test,
        y_pred=y_pred,
        feature_names=FEATURE_COLS,
        importances=model.feature_importances(),
    )


if __name__ == "__main__":
    main()
