"""
Définition de la VARIABLE CIBLE — le score de viralité.

Problème : les vues brutes ne sont pas comparables entre une vidéo de 2 ans et une
récente, ni entre une grosse chaîne et une petite, ni entre catégories/langues.

On définit donc la viralité comme une performance *relative et normalisée* :
  1. `views_per_day`   = vues / ancienneté en jours        (vitesse d'accumulation)
  2. `virality_log`    = log1p(views_per_day)              (compresse la longue traîne)
  3. `virality_score`  = z-score de `virality_log` au sein de chaque (catégorie × langue)
                         => une vidéo est "virale" *relativement à sa niche et sa langue*

C'est `virality_score` que le modèle de scoring (Sprint C) apprend à prédire (hypothèse H2),
à partir de features qui N'utilisent PAS les vues (cf. build_features.py — anti-fuite).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _days_since(published_at: pd.Series, *, now: pd.Timestamp | None = None) -> pd.Series:
    now = now or pd.Timestamp.now("UTC")
    dt = pd.to_datetime(published_at, utc=True, errors="coerce")
    days = (now - dt).dt.total_seconds() / 86400.0
    return days.clip(lower=1.0)  # éviter division par ~0 pour les vidéos très récentes


def compute_virality_target(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("category", "language"),
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Ajoute views_per_day, virality_log et virality_score (z-score par niche)."""
    out = df.copy()
    out["days_since_publish"] = _days_since(out["published_at"], now=now)
    views = pd.to_numeric(out.get("view_count"), errors="coerce").fillna(0)
    out["views_per_day"] = views / out["days_since_publish"]
    out["virality_log"] = np.log1p(out["views_per_day"])

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std and std > 0 else s * 0.0

    out["virality_score"] = (
        out.groupby(list(group_cols))["virality_log"].transform(_zscore)
    )
    return out
