"""
Auto-découverte des chaînes concurrentes à partir des données déjà collectées.

À partir de data/raw/competitors/videos.csv, classe les chaînes par présence
(nombre de vidéos + vues cumulées) et par catégorie, puis peut écrire/fusionner
le résultat dans config/channels.yaml — sans jamais inventer de handle.

Usage :
  python -m src.ingestion.discover_channels --top 3            # aperçu
  python -m src.ingestion.discover_channels --top 3 --write    # écrit dans channels.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "data" / "raw" / "competitors" / "videos.csv"
CHANNELS_YAML = ROOT / "config" / "channels.yaml"


def rank_channels(df: pd.DataFrame, *, top: int) -> pd.DataFrame:
    """Top chaînes par catégorie selon nb de vidéos puis vues cumulées."""
    g = (
        df.groupby(["category", "language", "channel_id", "channel_title"], dropna=True)
        .agg(videos=("video_id", "count"), total_views=("view_count", "sum"))
        .reset_index()
        .sort_values(["category", "videos", "total_views"], ascending=[True, False, False])
    )
    return g.groupby("category", group_keys=False).head(top)


def to_channel_entries(ranked: pd.DataFrame) -> list[dict]:
    entries, seen = [], set()
    for _, r in ranked.iterrows():
        cid = r["channel_id"]
        if not isinstance(cid, str) or cid in seen:
            continue
        seen.add(cid)
        entries.append({
            "id": cid,
            "category": r["category"],
            "language": r["language"],
            "note": f"{r['channel_title']} — {int(r['videos'])} vidéos, {int(r['total_views'])} vues (auto-découvert)",
        })
    return entries


def merge_into_yaml(entries: list[dict]) -> int:
    data = {}
    if CHANNELS_YAML.exists():
        data = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8")) or {}
    existing = data.get("channels") or []
    existing = [c for c in existing if isinstance(c, dict)]
    have = {c.get("id") for c in existing}
    added = [e for e in entries if e["id"] not in have]
    data["channels"] = existing + added
    CHANNELS_YAML.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return len(added)


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-découverte des chaînes concurrentes")
    p.add_argument("--top", type=int, default=3, help="chaînes par catégorie")
    p.add_argument("--write", action="store_true", help="fusionner dans config/channels.yaml")
    args = p.parse_args()

    if not VIDEOS.exists():
        raise SystemExit(f"Aucune donnée : lance d'abord la collecte (attendu : {VIDEOS}).")

    df = pd.read_csv(VIDEOS)
    ranked = rank_channels(df, top=args.top)
    entries = to_channel_entries(ranked)

    print(f"\n{len(entries)} chaînes proposées (top {args.top}/catégorie) :\n")
    for e in entries:
        print(f"  [{e['category']:<14} {e['language']}] {e['note']}")

    if args.write:
        n = merge_into_yaml(entries)
        print(f"\n→ {n} nouvelles chaînes ajoutées à {CHANNELS_YAML.relative_to(ROOT)}")
    else:
        print("\n(aperçu — ajoute --write pour mettre à jour channels.yaml)")


if __name__ == "__main__":
    main()
