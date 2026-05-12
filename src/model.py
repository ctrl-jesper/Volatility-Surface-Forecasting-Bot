"""
Machine learning models for implied volatility surface regression.
Author: Jesper Mathias Nielsen
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


class VolSurfaceModel:
    """
    Wraps a scikit-learn regressor for implied vol surface prediction.

    Supports Random Forest and Gradient Boosting backends.

    Parameters
    ----------
    model_type : str
        'rf' for Random Forest (default), 'gb' for Gradient Boosting.
    """

    _MODELS = {
        "rf": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42
        ),
        "gb": lambda: GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }

    def __init__(self, model_type: str = "rf") -> None:
        if model_type not in self._MODELS:
            raise ValueError(f"model_type must be one of {list(self._MODELS)}")
        self.model_type = model_type
        self.model = self._MODELS[model_type]()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the regression model on training data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate implied vol predictions."""
        return self.model.predict(X)

    def evaluate(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """
        Evaluate the model on held-out data.

        Returns
        -------
        tuple[float, float, np.ndarray]
            (MSE, R², predictions)
        """
        y_pred = self.predict(X)
        mse    = mean_squared_error(y_true, y_pred)
        r2     = r2_score(y_true, y_pred)
        return mse, r2, y_pred

    def feature_importances(self) -> np.ndarray:
        """Return feature importances from the underlying model."""
        return self.model.feature_importances_
