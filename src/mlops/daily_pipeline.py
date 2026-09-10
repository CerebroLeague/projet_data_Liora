"""
Boucle QUOTIDIENNE (inférence seule — aucun ré-entraînement).

Chaque matin :
  1. génère le menu du jour (idées FR + EN) pour chaque catégorie via le RAG
  2. score chaque idée avec le modèle du Sprint C, sélectionne le top-N
  3. écrit `data/processed/suggestions.json` (consommé par l'API / Streamlit)
  4. enregistre un résumé du run + vérifie la dérive (monitoring)

Conçu pour tourner en cron / GitHub Actions. En l'absence de `LLM_API_KEY`, bascule en mock.

Usage :
  python -m src.mlops.daily_pipeline               # toutes les catégories
  python -m src.mlops.daily_pipeline --category QUIZ_FOOT --n 5
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .. import taxonomy
from ..generator.generate import generate_ideas
from ..generator.retriever import Retriever
from . import drift

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("daily")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "suggestions.json"
HISTORY = ROOT / "reports" / "run_history.jsonl"


def _client():
    try:
        import os
        from ..generator.llm import LLMClient
        return LLMClient() if os.getenv("LLM_API_KEY") else None
    except Exception as e:
        log.warning("LLM indisponible (%s) → mode mock.", e)
        return None


def run(categories: list[str], n: int = 5) -> dict:
    load_dotenv(ROOT / ".env")
    retriever = Retriever()
    client = _client()
    mock = client is None
    log.info("Génération du menu (%s) — %d catégories × %d langues%s",
             "LLM réel" if not mock else "mock", len(categories), len(taxonomy.LANGUAGES),
             "" if not mock else " [MOCK]")

    menus: dict[str, dict] = {}
    all_scores: list[float] = []
    for cat in categories:
        menus[cat] = {}
        for lang in taxonomy.LANGUAGES:
            ideas = generate_ideas(cat, lang, n=n, retriever=retriever, client=client, mock=mock)
            menus[cat][lang] = ideas
            all_scores += [i.get("predicted_score", 0.0) for i in ideas]
        log.info("  %s ✓", cat)

    now = datetime.now(timezone.utc).isoformat()
    payload = {"generated_at": now, "llm_used": not mock, "menus": menus}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("→ %s", OUT.relative_to(ROOT))

    # résumé du run pour le monitoring
    summary = {
        "generated_at": now, "llm_used": not mock,
        "n_ideas": len(all_scores),
        "mean_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        "categories": categories,
    }
    _append_history(summary)
    flags = drift.check(HISTORY)
    for f in flags:
        log.warning("DÉRIVE : %s", f)
    return {"summary": summary, "drift": flags}


def _append_history(summary: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Boucle quotidienne (Sprint F)")
    p.add_argument("--category", nargs="*", default=taxonomy.CATEGORIES)
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()
    cats = [c for c in args.category if taxonomy.is_valid_category(c)]
    run(cats, n=args.n)


if __name__ == "__main__":
    main()
