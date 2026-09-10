"""
Orchestrateur du Sprint B — exploration & features.

Pipeline :
  1. Charge data/raw/competitors/videos.csv
  2. Calcule la cible de viralité (target.py)
  3. Construit les features anti-fuite (build_features.py) + garde-fou
  4. Sauvegarde le dataset modélisable dans data/processed/
  5. Génère les figures d'EDA dans reports/figures/ + un profil JSON

Usage :
  python -m src.features.run_features
  python -m src.features.run_features --no-figures
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path

import pandas as pd

from . import eda
from .build_features import assert_no_leakage, build_features, parse_duration_seconds
from .target import compute_virality_target

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("features")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "competitors" / "videos.csv"
PROCESSED = ROOT / "data" / "processed"


def _parse_tags(v):
    """Les tags sont sérialisés en str dans le CSV — on retente une liste Python."""
    if isinstance(v, str) and v.startswith("["):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    return v


def load_raw(path: Path = RAW) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Aucune donnée : lance d'abord la collecte (attendu : {path}).")
    df = pd.read_csv(path)
    if "tags" in df:
        df["tags"] = df["tags"].apply(_parse_tags)
    return df


def run(make_figures: bool = True) -> None:
    df = load_raw()
    log.info("Chargé %d vidéos brutes.", len(df))

    labeled = compute_virality_target(df)
    feats = build_features(labeled)
    assert_no_leakage(feats)
    log.info("Features construites : %d colonnes (anti-fuite ✓).", feats.shape[1])

    # dataset modélisable = features + cible (alignées par video_id)
    target = labeled[["video_id", "virality_score", "views_per_day"]]
    dataset = feats.merge(target, on="video_id", how="left")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(PROCESSED / "dataset.csv", index=False)
    log.info("→ data/processed/dataset.csv (%d lignes, %d colonnes)", *dataset.shape)

    # profil JSON
    prof = eda.profile(labeled)
    (PROCESSED / "profile.json").write_text(
        json.dumps(prof, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("→ data/processed/profile.json")

    if make_figures:
        labeled["duration_seconds"] = labeled["duration_iso"].apply(parse_duration_seconds)
        paths = eda.generate_all(labeled)
        log.info("→ %d figures dans reports/figures/", len(paths))
        for p in paths:
            log.info("   %s", p.relative_to(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Exploration & features (Sprint B)")
    p.add_argument("--no-figures", action="store_true", help="ne pas générer les figures")
    args = p.parse_args()
    run(make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
