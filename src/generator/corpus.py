"""
Corpus documentaire du RAG — construit à partir des données déjà collectées.

Trois sources de "grounding", toutes étiquetées (catégorie × langue) :
  1. STYLE   : titres de vidéos concurrentes qui marchent (exemples de formulation)
  2. TENDANCE: sujets Google Trends (fraîcheur / actualité)
  3. MOTS-CLÉS: termes porteurs par niche
(+ optionnel) corpus interne de quiz si présent dans data/raw/internal/.

Chaque document = dict {id, text, category, language, source, meta}.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_documents() -> list[dict]:
    docs: list[dict] = []

    # 1. STYLE — titres concurrents (avec vues pour trier par succès)
    vids = _read_csv(RAW / "competitors" / "videos.csv")
    for _, r in vids.iterrows():
        title = str(r.get("title") or "").strip()
        if not title:
            continue
        docs.append({
            "id": f"vid:{r.get('video_id')}",
            "text": title,
            "category": r.get("category"),
            "language": r.get("language"),
            "source": "style",
            "meta": {"views": int(r["view_count"]) if pd.notna(r.get("view_count")) else 0,
                     "family": r.get("family"), "channel": r.get("channel_title")},
        })

    # 2. TENDANCE — requêtes Google Trends
    trends = _read_csv(RAW / "trends" / "trends.csv")
    for _, r in trends.iterrows():
        q = str(r.get("query") or "").strip()
        if not q:
            continue
        docs.append({
            "id": f"trend:{r.get('category')}:{r.get('language')}:{q}",
            "text": q,
            "category": r.get("category"),
            "language": r.get("language"),
            "source": "trend",
            "meta": {"trend_type": r.get("trend_type"), "value": r.get("value")},
        })

    # 3. MOTS-CLÉS — termes agrégés
    kws = _read_csv(RAW / "keywords" / "keywords.csv")
    for _, r in kws.iterrows():
        term = str(r.get("term") or "").strip()
        if not term:
            continue
        docs.append({
            "id": f"kw:{r.get('category')}:{r.get('language')}:{term}",
            "text": term,
            "category": r.get("category"),
            "language": r.get("language"),
            "source": "keyword",
            "meta": {"freq": r.get("freq")},
        })

    # 4. CORPUS INTERNE (optionnel) — fichiers texte dans data/raw/internal/
    internal = RAW / "internal"
    if internal.exists():
        for f in internal.glob("*.txt"):
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if line:
                    docs.append({
                        "id": f"internal:{f.stem}:{i}",
                        "text": line, "category": None, "language": None,
                        "source": "internal", "meta": {"file": f.name},
                    })

    return docs


def stats(docs: list[dict]) -> dict:
    df = pd.DataFrame(docs)
    return {
        "total": len(docs),
        "by_source": df["source"].value_counts().to_dict() if not df.empty else {},
    }
