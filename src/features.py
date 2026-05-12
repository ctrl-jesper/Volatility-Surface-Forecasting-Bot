"""
Feature engineering for implied volatility surface modelling.
Author: Jesper Mathias Nielsen
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive analytical features from raw options data.

    Features added:
    - moneyness         : strike / spot  (1.0 = at-the-money)
    - log_moneyness     : log(strike / spot)  — symmetric around ATM
    - time_to_maturity  : direct pass-through of the maturity column
    - sqrt_maturity     : square-root of maturity (vol scales with √T)
    - historical_volatility : term-structure proxy (decays with maturity)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: strike, spot, maturity.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional feature columns appended.
    """
    df = df.copy()
    df["moneyness"]           = df["strike"] / df["spot"]
    df["log_moneyness"]       = np.log(df["strike"] / df["spot"])
    df["time_to_maturity"]    = df["maturity"]
    df["sqrt_maturity"]       = np.sqrt(df["maturity"])
    df["historical_volatility"] = 0.18 + 0.05 * np.exp(-df["maturity"])
    return df


FEATURE_COLS = [
    "moneyness",
    "log_moneyness",
    "time_to_maturity",
    "sqrt_maturity",
    "historical_volatility",
]
