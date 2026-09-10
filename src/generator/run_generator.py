"""
Orchestrateur du Sprint D — génération RAG d'idées de quiz.

Exemples :
  # test hors-ligne (sans LLM ni clé) — valide retrieval + scoring
  python -m src.generator.run_generator --category QUIZ_FOOT --language FR --mock

  # génération réelle (nécessite LLM_API_KEY dans .env)
  python -m src.generator.run_generator --category QUIZ_KPOP --language EN --n 5

  # "menu du jour" : n idées FR + n idées EN, sauvegardé en JSON
  python -m src.generator.run_generator --menu --category QUIZ_FOOT --n 5 --save
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from .. import taxonomy
from .generate import generate_ideas
from .retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("generator")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "suggestions.json"


def _client(mock: bool):
    if mock:
        return None
    try:
        from .llm import LLMClient
        return LLMClient()
    except Exception as e:
        log.warning("LLM indisponible (%s) → bascule en mode --mock.", e)
        return None


def _print(category: str, language: str, ideas: list[dict]) -> None:
    print(f"\n=== {taxonomy.CATEGORY_LABELS.get(category, category)} [{language}] ===")
    for rank, it in enumerate(ideas, 1):
        fam = taxonomy.family_label(it["family"]) if it.get("family") else "—"
        print(f"  #{rank}  score={it.get('predicted_score'):>6}  [{fam:<16}] {it['title']}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description="Génération RAG d'idées de quiz (Sprint D)")
    p.add_argument("--category", required=True, choices=taxonomy.CATEGORIES)
    p.add_argument("--language", default="FR", choices=taxonomy.LANGUAGES)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--theme", default=None, help="thème libre (recherche sémantique)")
    p.add_argument("--menu", action="store_true", help="générer FR + EN (menu du jour)")
    p.add_argument("--mock", action="store_true", help="sans LLM (test retrieval + scoring)")
    p.add_argument("--save", action="store_true", help="sauvegarder en data/processed/suggestions.json")
    args = p.parse_args()

    retriever = Retriever()  # index construit une fois
    client = _client(args.mock)
    langs = taxonomy.LANGUAGES if args.menu else [args.language]

    menu = {}
    for lang in langs:
        ideas = generate_ideas(args.category, lang, n=args.n, theme=args.theme,
                               retriever=retriever, client=client, mock=args.mock)
        _print(args.category, lang, ideas)
        menu[lang] = ideas

    if args.save:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"category": args.category, "menu": menu},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("→ %s", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
