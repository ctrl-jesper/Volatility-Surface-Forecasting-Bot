"""
Options data simulation and loading for the Volatility Surface Forecasting model.
Author: Jesper Mathias Nielsen
"""

import numpy as np
import pandas as pd


def simulate_options_data(n_samples: int = 1200, spot: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """
    Simulate realistic implied volatility data across strikes and maturities.

    The smile shape is modelled as a function of moneyness and maturity:
    - Higher vol for deep OTM strikes (skew)
    - Higher vol for shorter maturities (term structure inversion)
    - Small random noise to mimic market imperfections

    Parameters
    ----------
    n_samples : int
        Number of synthetic option data points to generate.
    spot : float
        Reference spot price of the underlying.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: strike, maturity, implied_volatility, spot
    """
    rng = np.random.default_rng(seed)

    strikes    = rng.uniform(75, 125, n_samples)
    maturities = rng.uniform(0.05, 2.0, n_samples)

    moneyness = strikes / spot - 1  # centred around 0 at ATM

    # Volatility smile: U-shaped in moneyness, term-structure decay
    base_vol  = 0.18
    smile     = 0.20 * moneyness ** 2          # symmetric smile
    skew      = -0.08 * moneyness               # negative skew (left tail premium)
    term_str  = 0.06 * np.exp(-maturities)     # short-end elevation
    noise     = rng.normal(0, 0.008, n_samples)

    implied_vols = np.clip(base_vol + smile + skew + term_str + noise, 0.05, 0.80)

    return pd.DataFrame({
        "strike":            strikes,
        "maturity":          maturities,
        "implied_volatility": implied_vols,
        "spot":              spot,
    })


def load_data(filepath: str | None = None) -> pd.DataFrame:
    """
    Load options data from CSV or fall back to synthetic simulation.

    CSV must contain columns: strike, maturity, implied_volatility, spot.

    Parameters
    ----------
    filepath : str or None
        Path to a real options CSV file. If None, synthetic data is used.

    Returns
    -------
    pd.DataFrame
    """
    if filepath:
        return pd.read_csv(filepath)
    return simulate_options_data()
