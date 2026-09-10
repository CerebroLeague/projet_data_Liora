"""
Modèles de scoring viral : baselines → LightGBM optimisé.

Stratégie (conforme à la grille DataScientest) :
  1. Baseline triviale  : DummyRegressor (moyenne)        -> plancher d'erreur
  2. Baseline simple    : régression linéaire             -> référence "raisonnable"
  3. Modèle avancé      : LightGBM (boosting)             -> le modèle visé
  4. Optimisation       : GridSearchCV + validation croisée (scorer = Spearman)
"""
from __future__ import annotations

from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import spearman_scorer


def make_baselines() -> dict:
    """Modèles de référence."""
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        # standardisation utile pour la régression linéaire
        "linear": Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
    }


def make_lgbm() -> LGBMRegressor:
    """LightGBM par défaut (paramètres prudents vu le petit volume ~476 lignes)."""
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def grid_search(X, y, *, cv: int = 5) -> GridSearchCV:
    """Optimise LightGBM par validation croisée, en maximisant Spearman."""
    grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [5, 10, 20],
    }
    base = LGBMRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    gs = GridSearchCV(base, grid, scoring=spearman_scorer, cv=cv, n_jobs=1, refit=True)
    gs.fit(X, y)
    return gs
