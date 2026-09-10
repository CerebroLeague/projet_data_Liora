"""
Chargement du dataset modélisable et séparation features / cible.

Garde-fou anti-fuite : on retire explicitement `views_per_day` (base de la cible) et
l'identifiant `video_id`. La cible est `virality_score`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "dataset.csv"

TARGET = "virality_score"
DROP = ["video_id", "virality_score", "views_per_day"]  # non-features (dont fuite)


def load_xy(path: Path | str = DATASET) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Dataset absent : lance d'abord `python -m src.features.run_features` ({path}).")
    df = pd.read_csv(path)
    y = df[TARGET].astype(float)
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    # les colonnes one-hot peuvent être lues en bool/str -> on force en numérique
    X = X.replace({True: 1, False: 0, "True": 1, "False": 0}).apply(pd.to_numeric, errors="coerce")
    return X, y


def split(X: pd.DataFrame, y: pd.Series, *, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=seed)
