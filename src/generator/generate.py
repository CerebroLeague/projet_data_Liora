"""
Génération d'idées de quiz par RAG :
  contexte récupéré (retriever)  →  prompt  →  LLM  →  idées JSON  →  scoring  →  classement.
"""
from __future__ import annotations

import json
import re

from .. import taxonomy
from .retriever import Retriever
from .scorer import IdeaScorer

# familles proposables (actives en production)
_FAMILIES_HINT = ", ".join(f"{c} ({taxonomy.family_label(c)})" for c in taxonomy.ACTIVE_FAMILIES)

SYSTEM = (
    "Tu es un expert en création de quiz vidéo courts et viraux pour YouTube, bilingue FR/EN. "
    "Tu génères des IDÉES de quiz nichées, originales et à fort potentiel viral. "
    "Tu réponds UNIQUEMENT par un tableau JSON valide, sans texte autour."
)


def _user_prompt(category: str, language: str, n: int, ctx: dict) -> str:
    lang_name = "français" if language == "FR" else "anglais"
    theme = taxonomy.CATEGORY_LABELS.get(category, category)
    examples = "\n".join(f"- {t}" for t in ctx["examples"][:8]) or "(aucun)"
    trends = ", ".join(ctx["trends"][:6]) or "(aucune)"
    keywords = ", ".join(ctx["keywords"][:10]) or "(aucun)"
    return f"""Génère {n} idées de quiz en {lang_name} pour la catégorie « {theme} ».

Titres de vidéos concurrentes qui MARCHENT (inspire-toi du style, ne copie pas) :
{examples}

Sujets tendance du moment : {trends}
Mots-clés porteurs : {keywords}

Contraintes :
- Chaque idée doit choisir une FAMILLE parmi : {_FAMILIES_HINT}
- Titres courts, accrocheurs, en {lang_name}.
- Idées variées (familles différentes), nichées et originales.

Réponds par un tableau JSON de {n} objets, chacun ainsi :
{{"title": "...", "family": "CODE_FAMILLE", "description": "1 phrase", "why": "pourquoi ça peut être viral"}}"""


def _parse_json_array(text: str) -> list[dict]:
    """Extrait le premier tableau JSON du texte (robuste aux entourages)."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return [d for d in data if isinstance(d, dict) and d.get("title")]
    except json.JSONDecodeError:
        return []


def _mock_ideas(category: str, language: str, n: int, ctx: dict) -> list[dict]:
    """Génère des idées factices depuis le contexte (test hors-ligne, sans LLM)."""
    fams = taxonomy.ACTIVE_FAMILIES
    base = ctx["examples"] or ctx["trends"] or ["Idée de quiz"]
    out = []
    for i in range(n):
        out.append({
            "title": f"[MOCK] {base[i % len(base)][:60]}",
            "family": fams[i % len(fams)],
            "description": "Idée générée en mode mock (sans LLM).",
            "why": "Basée sur un exemple à succès récupéré par le RAG.",
        })
    return out


def generate_ideas(category: str, language: str, *, n: int = 5, theme: str | None = None,
                   retriever: Retriever | None = None, client=None, mock: bool = False) -> list[dict]:
    """Pipeline complet RAG → idées scorées et classées."""
    retriever = retriever or Retriever()
    ctx = retriever.build_context(category, language, theme=theme)

    if mock or client is None:
        ideas = _mock_ideas(category, language, n, ctx)
    else:
        raw = client.chat(SYSTEM, _user_prompt(category, language, n, ctx))
        ideas = _parse_json_array(raw)

    # nettoyage famille (garder un code valide, sinon None)
    for it in ideas:
        fam = str(it.get("family") or "").upper()
        it["family"] = fam if fam in taxonomy.FAMILY_CODES else None

    return IdeaScorer().score(ideas, category=category, language=language)
