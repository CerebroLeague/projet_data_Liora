"""
Métriques d'évaluation du scoring viral.

Métrique PRINCIPALE : corrélation de Spearman (rang) — on veut surtout bien *classer*
les idées par potentiel viral (hypothèse H2 : Spearman > 0,5).
Métriques secondaires : MAE, RMSE (erreur sur la valeur du z-score).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error


def spearman(y_true, y_pred) -> float:
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if rho == rho else 0.0  # NaN (prédiction constante) -> 0


# scorer utilisable par GridSearchCV (plus c'est haut, mieux c'est)
spearman_scorer = make_scorer(spearman, greater_is_better=True)


def evaluate(y_true, y_pred) -> dict:
    return {
        "spearman": round(spearman(y_true, y_pred), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }
