"""
Visualisation for the Volatility Surface Forecasting model.
Author: Jesper Mathias Nielsen
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def plot_all_vertical(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str] | None = None,
    importances: np.ndarray | None = None,
) -> None:
    """
    Render a three-panel dark-themed figure:
    1. Predicted 3-D volatility surface
    2. Actual vs Predicted scatter
    3. Feature importance bar chart (if importances supplied)

    Parameters
    ----------
    X, Y : np.ndarray
        Meshgrid arrays for strike and maturity axes.
    Z : np.ndarray
        Predicted implied vol on the grid (reshaped to X.shape).
    y_true, y_pred : np.ndarray
        Hold-out actual and predicted values.
    feature_names : list[str] or None
        Feature labels for the importance chart.
    importances : np.ndarray or None
        Feature importance values from the fitted model.
    """
    plt.style.use("dark_background")
    n_panels = 3 if (importances is not None and feature_names is not None) else 2
    fig = plt.figure(figsize=(10, 5 * n_panels))
    fig.suptitle("Volatility Surface Forecasting", fontsize=14, color="white", y=1.01)

    # ── Panel 1: 3-D surface ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(n_panels, 1, 1, projection="3d")
    surf = ax1.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", antialiased=True)
    ax1.set_xlabel("Strike",            fontsize=10)
    ax1.set_ylabel("Maturity (Years)",  fontsize=10)
    ax1.set_zlabel("Implied Vol",       fontsize=10)
    ax1.set_title("Predicted Volatility Surface", fontsize=12)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1)

    # ── Panel 2: Actual vs predicted ─────────────────────────────────────────
    ax2 = fig.add_subplot(n_panels, 1, 2)
    ax2.scatter(y_true, y_pred, alpha=0.55, c="cyan", edgecolors="black", linewidths=0.4, s=18)
    lim = [min(y_true.min(), y_pred.min()) - 0.01, max(y_true.max(), y_pred.max()) + 0.01]
    ax2.plot(lim, lim, "r--", lw=1.8, label="Ideal (y = x)")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("Actual Implied Vol",    fontsize=10)
    ax2.set_ylabel("Predicted Implied Vol", fontsize=10)
    ax2.set_title("Actual vs Predicted",    fontsize=12)
    ax2.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax2.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # ── Panel 3: Feature importance ───────────────────────────────────────────
    if n_panels == 3:
        ax3 = fig.add_subplot(n_panels, 1, 3)
        idx = np.argsort(importances)
        ax3.barh(
            [feature_names[i] for i in idx],
            importances[idx],
            color="steelblue",
        )
        ax3.set_title("Feature Importances", fontsize=12)
        ax3.set_xlabel("Importance")
        ax3.grid(True, axis="x", color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()
