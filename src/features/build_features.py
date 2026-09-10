"""
Feature engineering pour le scoring viral — SANS FUITE de la cible.

Principe anti-fuite : la cible (virality_score) dérive des VUES. Le modèle prédictif doit
donc estimer le potentiel viral d'une vidéo *avant publication*, à partir de features
disponibles a priori : titre, durée, catégorie, famille, langue, moment de publication,
signaux de tendance / mots-clés. => On EXCLUT view_count, like_count, comment_count.

Familles de features :
  - texte du titre (longueur, mots, chiffres, question, emoji, MAJUSCULES)
  - format vidéo (durée, short ou non)
  - temporel (jour de semaine, heure de publication)
  - axes catégoriels (catégorie, famille, family_group, langue)
  - tags (nombre)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# colonnes interdites en entrée du modèle (fuite directe de la cible)
LEAKAGE_COLS = ["view_count", "like_count", "comment_count", "views_per_day",
                "virality_log", "virality_score"]


def parse_duration_seconds(iso: str | float | None) -> float:
    """Convertit une durée ISO8601 (PT#H#M#S) en secondes."""
    if not isinstance(iso, str):
        return np.nan
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return np.nan
    h, mn, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mn * 60 + s


def _title_features(title: str) -> dict:
    t = title or ""
    words = _WORD_RE.findall(t)
    letters = [c for c in t if c.isalpha()]
    return {
        "title_len_chars": len(t),
        "title_word_count": len(words),
        "title_has_number": int(bool(re.search(r"\d", t))),
        "title_has_question": int("?" in t),
        "title_has_emoji": int(bool(_EMOJI_RE.search(t))),
        "title_upper_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "title_exclam": int("!" in t),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DataFrame de features prêtes pour le modèle.
    Conserve `video_id` (clé) et les axes catégoriels encodés en dummies.
    """
    rows = []
    published = pd.to_datetime(df.get("published_at"), utc=True, errors="coerce")
    for i, (_, r) in enumerate(df.iterrows()):
        dur = parse_duration_seconds(r.get("duration_iso"))
        ts = published.iloc[i]
        tags = r.get("tags")
        n_tags = len(tags) if isinstance(tags, (list, tuple)) else (
            len(str(tags).split(",")) if isinstance(tags, str) and tags.strip() else 0
        )
        feat = {
            "video_id": r.get("video_id"),
            **_title_features(r.get("title")),
            "duration_seconds": dur,
            "is_short": int(dur <= 60) if not np.isnan(dur) else 0,
            "tag_count": n_tags,
            "publish_dow": ts.dayofweek if pd.notna(ts) else np.nan,      # 0=lundi
            "publish_hour": ts.hour if pd.notna(ts) else np.nan,
            "category": r.get("category"),
            "family": r.get("family") or "UNKNOWN",
            "family_group": r.get("family_group") or "Autre",
            "language": r.get("language"),
        }
        rows.append(feat)

    feats = pd.DataFrame(rows)
    # encodage one-hot des axes catégoriels
    feats = pd.get_dummies(
        feats, columns=["category", "family", "family_group", "language"],
        prefix=["cat", "fam", "famgrp", "lang"], dummy_na=False,
    )
    return feats


def assert_no_leakage(feats: pd.DataFrame) -> None:
    """Garde-fou : échoue si une colonne dérivée des vues s'est glissée dans les features."""
    leaked = [c for c in LEAKAGE_COLS if c in feats.columns]
    if leaked:
        raise ValueError(f"Fuite de la cible détectée dans les features : {leaked}")
