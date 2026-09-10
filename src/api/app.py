"""
API d'inférence (FastAPI) — point d'intégration du système dans une application externe.

Charge les modèles UNE fois au démarrage (jamais de ré-entraînement en ligne).

Endpoints :
  GET  /health              -> état du service
  POST /generate            -> génère des idées de quiz (RAG) pour une catégorie × langue
  POST /score               -> score des idées fournies avec le modèle de scoring
  GET  /suggestions/today   -> "menu du jour" pré-calculé (ou généré en repli)

Lancement :
  uvicorn src.api.app:app --reload
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .. import taxonomy
from ..generator.generate import generate_ideas
from ..generator.retriever import Retriever
from ..generator.scorer import IdeaScorer

ROOT = Path(__file__).resolve().parents[2]
SUGGESTIONS = ROOT / "data" / "processed" / "suggestions.json"

load_dotenv(ROOT / ".env")

# ── état partagé (chargé une fois au démarrage) ────────────────
STATE: dict = {}


def _try_llm():
    try:
        from ..generator.llm import LLMClient
        return LLMClient() if os.getenv("LLM_API_KEY") else None
    except Exception:
        return None


def _load():
    STATE["retriever"] = Retriever()
    STATE["scorer"] = IdeaScorer()
    STATE["llm"] = _try_llm()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()          # chargement des modèles au démarrage
    yield


app = FastAPI(title="Quiz IA — API", version="1.0", lifespan=lifespan)

# ── CORS ───────────────────────────────────────────────────────
# Autorise le frontend (navigateur) à appeler l'API. Origines surchargées
# via la variable d'env CORS_ORIGINS (liste séparée par des virgules).
_cors_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:3003,http://localhost:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schémas ────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    category: str = Field(..., description="code catégorie, ex QUIZ_FOOT")
    language: str = "FR"
    n: int = 5
    theme: str | None = None


class Idea(BaseModel):
    title: str
    family: str | None = None
    description: str | None = None


class ScoreRequest(BaseModel):
    category: str
    language: str = "FR"
    ideas: list[Idea]


# ── endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "scorer" in STATE, "llm": STATE.get("llm") is not None,
            "categories": taxonomy.CATEGORIES, "languages": taxonomy.LANGUAGES}


@app.post("/generate")
def generate(req: GenerateRequest):
    if not taxonomy.is_valid_category(req.category):
        raise HTTPException(400, f"catégorie invalide: {req.category}")
    ideas = generate_ideas(
        req.category, req.language, n=req.n, theme=req.theme,
        retriever=STATE["retriever"], client=STATE.get("llm"),
        mock=STATE.get("llm") is None,
    )
    return {"category": req.category, "language": req.language,
            "llm_used": STATE.get("llm") is not None, "ideas": ideas}


@app.post("/score")
def score(req: ScoreRequest):
    ideas = [i.model_dump() for i in req.ideas]
    scored = STATE["scorer"].score(ideas, category=req.category, language=req.language)
    return {"category": req.category, "language": req.language, "ideas": scored}


@app.get("/suggestions/today")
def suggestions_today(category: str = "QUIZ_FOOT", n: int = 5):
    """Menu du jour pré-calculé (boucle quotidienne) si dispo, sinon génération en repli."""
    if SUGGESTIONS.exists():
        data = json.loads(SUGGESTIONS.read_text(encoding="utf-8"))
        menus = data.get("menus", {})
        if category in menus:
            return {"category": category, "generated_at": data.get("generated_at"),
                    "precomputed": True, "menu": menus[category]}
    # repli : génération à la volée
    menu = {}
    for lang in taxonomy.LANGUAGES:
        menu[lang] = generate_ideas(category, lang, n=n, retriever=STATE["retriever"],
                                    client=STATE.get("llm"), mock=STATE.get("llm") is None)
    return {"category": category, "menu": menu, "precomputed": False}
