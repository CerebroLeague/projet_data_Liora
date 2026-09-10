"""
Orchestrateur du Sprint A — ingestion & étiquetage.

Pipeline :
  1. Découverte de vidéos concurrentes par mots-clés (seeds × catégories × langues)
  2. Vidéos des chaînes connues (config/channels.yaml)
  3. Récupération des détails (statistiques) puis étiquetage 3 axes
  4. Sujets tendance (Google Trends) par catégorie
  5. Extraction de mots-clés agrégés par (catégorie × langue)
  6. Écriture dans data/raw/{competitors,trends,keywords}

Exemples :
  python -m src.ingestion.run_ingestion --categories QUIZ_FOOT QUIZ_KPOP --languages FR
  python -m src.ingestion.run_ingestion --max-per-query 10 --skip-trends
  python -m src.ingestion.run_ingestion --dry-run        # n'appelle aucune API
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from .. import taxonomy
from . import keywords as kw
from .labeler import build_category_matcher, label_record
from .youtube_collector import QuotaExceeded, YouTubeCollector

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("ingestion")

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
RAW = ROOT / "data" / "raw"


# ── config ────────────────────────────────────────────────────
def load_seeds() -> dict[str, dict[str, list[str]]]:
    with open(CONFIG / "categories_keywords.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_channels() -> list[dict]:
    with open(CONFIG / "channels.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    chans = data.get("channels") or []
    return [c for c in chans if isinstance(c, dict)]


def save(df: pd.DataFrame, subdir: str, name: str) -> Path:
    out_dir = RAW / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_csv(path, index=False)
    log.info("→ %s (%d lignes)", path.relative_to(ROOT), len(df))
    return path


# ── collecte vidéos ───────────────────────────────────────────
def collect_competitors(args, seeds, matcher) -> list[dict]:
    yt = YouTubeCollector()
    pending: list[tuple[str, str | None, str | None]] = []  # (video_id, cat_hint, lang_hint)

    # 1) découverte par mots-clés
    for category in args.categories:
        by_lang = seeds.get(category, {})
        for lang in args.languages:
            for seed in by_lang.get(lang, []):
                loc = taxonomy.LANGUAGE_LOCALE[lang]
                try:
                    ids = yt.search_by_query(
                        seed,
                        region_code=loc["region_code"],
                        relevance_language=loc["relevance_language"],
                        max_results=args.max_per_query,
                        published_after=args.published_after,
                    )
                except QuotaExceeded as e:
                    log.error("%s — arrêt de la recherche.", e)
                    ids = []
                    args.categories = []  # stoppe les boucles suivantes proprement
                log.info("search '%s' [%s/%s] → %d vidéos (quota≈%d)",
                         seed, category, lang, len(ids), yt.quota_used)
                pending += [(vid, category, lang) for vid in ids]

    # 2) chaînes connues
    for ch in load_channels():
        cid = yt.resolve_channel_id(handle=ch.get("handle"), channel_id=ch.get("id"))
        if not cid:
            log.warning("Chaîne non résolue : %s", ch)
            continue
        ids = yt.videos_of_channel(cid, max_results=args.max_channel_videos)
        log.info("chaîne %s → %d vidéos", ch.get("handle") or cid, len(ids))
        pending += [(vid, ch.get("category"), ch.get("language")) for vid in ids]

    # 3) détails + étiquetage (dédoublonné par video_id)
    hints = {vid: (cat, lang) for vid, cat, lang in pending}
    records: list[dict] = []
    try:
        for rec in yt.video_details(hints.keys()):
            cat_hint, lang_hint = hints.get(rec["video_id"], (None, None))
            records.append(label_record(rec, category_hint=cat_hint,
                                        language_hint=lang_hint, matcher=matcher))
    except QuotaExceeded as e:
        log.error("%s — on sauvegarde ce qui a été récupéré.", e)

    log.info("Total vidéos étiquetées : %d (quota≈%d)", len(records), yt.quota_used)
    return records


# ── trends ────────────────────────────────────────────────────
def collect_trends(args, seeds) -> list[dict]:
    from .trends_collector import TrendsCollector
    subset = {c: {l: seeds.get(c, {}).get(l, []) for l in args.languages}
              for c in args.categories}
    return TrendsCollector(sleep=args.trends_sleep).collect(subset)


# ── main ──────────────────────────────────────────────────────
def main() -> None:
    load_dotenv(ROOT / ".env")
    seeds = load_seeds()

    p = argparse.ArgumentParser(description="Ingestion & étiquetage (Sprint A)")
    p.add_argument("--categories", nargs="*", default=taxonomy.CATEGORIES)
    p.add_argument("--languages", nargs="*", default=taxonomy.LANGUAGES)
    p.add_argument("--max-per-query", type=int, default=15, help="vidéos max par seed de recherche")
    p.add_argument("--max-channel-videos", type=int, default=50)
    p.add_argument("--published-after", default=None, help="ISO8601, ex 2024-01-01T00:00:00Z")
    p.add_argument("--skip-trends", action="store_true")
    p.add_argument("--trends-sleep", type=float, default=2.0)
    p.add_argument("--dry-run", action="store_true", help="valide la config sans appeler d'API")
    args = p.parse_args()

    # validation des axes
    args.categories = [c for c in args.categories if taxonomy.is_valid_category(c)]
    args.languages = [l for l in args.languages if taxonomy.is_valid_language(l)]
    log.info("Catégories: %s | Langues: %s", args.categories, args.languages)

    if args.dry_run:
        n_seeds = sum(len(seeds.get(c, {}).get(l, []))
                      for c in args.categories for l in args.languages)
        log.info("DRY-RUN ✓ %d seeds, %d chaînes configurées, %d catégories valides.",
                 n_seeds, len(load_channels()), len(args.categories))
        return

    matcher = build_category_matcher(seeds)

    # 1-3 vidéos concurrentes
    records = collect_competitors(args, seeds, matcher)
    if records:
        save(pd.DataFrame(records), "competitors", "videos.csv")
        # 5 mots-clés
        save(pd.DataFrame(kw.extract_keywords(records)), "keywords", "keywords.csv")

    # 4 trends
    if not args.skip_trends:
        trends = collect_trends(args, seeds)
        if trends:
            save(pd.DataFrame(trends), "trends", "trends.csv")


if __name__ == "__main__":
    main()
