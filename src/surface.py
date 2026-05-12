"""
Grid utilities for implied volatility surface construction.
Author: Jesper Mathias Nielsen
"""

import numpy as np
import pandas as pd


def create_surface_grid(
    strike_range: tuple[float, float],
    maturity_range: tuple[float, float],
    spot: float,
    n_strikes: int = 60,
    n_maturities: int = 60,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build a dense (strike × maturity) evaluation grid for surface rendering.

    Parameters
    ----------
    strike_range : tuple[float, float]
        (min_strike, max_strike) bounds for the grid.
    maturity_range : tuple[float, float]
        (min_maturity, max_maturity) in years.
    spot : float
        Current spot price (used for moneyness features).
    n_strikes : int
        Number of grid points along the strike axis.
    n_maturities : int
        Number of grid points along the maturity axis.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray]
        - grid_df : flat DataFrame ready for feature engineering and prediction
        - grid_strike : 2-D meshgrid array for plotting
        - grid_maturity : 2-D meshgrid array for plotting
    """
    strikes    = np.linspace(strike_range[0],   strike_range[1],   n_strikes)
    maturities = np.linspace(maturity_range[0], maturity_range[1], n_maturities)

    grid_strike, grid_maturity = np.meshgrid(strikes, maturities)

    grid_df = pd.DataFrame({
        "strike":   grid_strike.ravel(),
        "maturity": grid_maturity.ravel(),
        "spot":     spot,
    })

    return grid_df, grid_strike, grid_maturity
