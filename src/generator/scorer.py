"""
Scoring des idées générées, avec le MÊME modèle que le Sprint C.

On reconstruit les features d'une idée (titre + catégorie + langue + famille) via
`build_features`, on aligne les colonnes sur celles attendues par le modèle
(`models/feature_columns.json`), puis on prédit le `virality_score`.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .. import taxonomy
from ..features.build_features import build_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "scoring_viral.joblib"
COLS_PATH = ROOT / "models" / "feature_columns.json"


class IdeaScorer:
    def __init__(self):
        if not MODEL_PATH.exists() or not COLS_PATH.exists():
            raise SystemExit("Modèle de scoring absent : lance d'abord `python -m src.scoring.run_scoring`.")
        self.model = joblib.load(MODEL_PATH)
        self.columns = json.loads(COLS_PATH.read_text(encoding="utf-8"))

    def score(self, ideas: list[dict], *, category: str, language: str) -> list[dict]:
        """Ajoute `predicted_score` à chaque idée et retourne la liste triée décroissante."""
        if not ideas:
            return []
        rows = []
        for it in ideas:
            fam = it.get("family")
            rows.append({
                "video_id": None,
                "title": it.get("title"),
                "description": it.get("description"),
                "published_at": None, "tags": [], "duration_iso": None,
                "category": category, "language": language,
                "family": fam,
                "family_group": taxonomy.family_group(fam) if fam else None,
            })
        feats = build_features(pd.DataFrame(rows))
        feats = feats.drop(columns=["video_id"], errors="ignore")
        # aligner sur le schéma d'entraînement (colonnes manquantes = 0)
        feats = feats.reindex(columns=self.columns, fill_value=0)
        preds = self.model.predict(feats)
        for it, p in zip(ideas, preds):
            it["predicted_score"] = round(float(p), 3)
        return sorted(ideas, key=lambda x: x["predicted_score"], reverse=True)
