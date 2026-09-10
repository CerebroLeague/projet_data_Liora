"""
Étiquetage des données scrapées sur les 3 axes du projet :
  - LANGUE   : détection FR/EN (langdetect, avec repli sur le contexte de collecte)
  - CATÉGORIE: connue par la seed de découverte, sinon inférée par mots-clés
  - FAMILLE  : inférée par règles sur le titre (les vidéos concurrentes ne sont pas pré-taguées)

L'inférence est volontairement simple et transparente (règles) : c'est un point de départ
documentable dans le rapport ; elle pourra être remplacée par un sous-modèle de classification.
"""
from __future__ import annotations

import re

from .. import taxonomy

# ── Langue ────────────────────────────────────────────────────
try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0  # détection déterministe
except Exception:  # pragma: no cover
    detect = None


def detect_language(text: str, *, fallback: str = "FR") -> str:
    """Retourne 'FR' ou 'EN'. Repli sur `fallback` (langue de la seed de collecte)."""
    if detect and text and text.strip():
        try:
            code = detect(text)
            if code == "fr":
                return "FR"
            if code == "en":
                return "EN"
        except Exception:
            pass
    return fallback if fallback in taxonomy.LANGUAGES else "FR"


# ── Famille (règles sur le titre + description) ───────────────
# Ordre = priorité : la première famille dont un motif matche gagne.
_FAMILY_PATTERNS: list[tuple[str, list[str]]] = [
    ("COMPO_DRAPEAUX",     [r"compo.*drapeau", r"drapeau.*compo", r"lineup.*flag", r"flag.*lineup", r"composition.*drapeau"]),
    ("REBUS_EMOJIS",       [r"emoji", r"rébus", r"rebus", r"guess.*emoji"]),
    ("TU_PREFERES",        [r"tu pr[ée]f[èe]res", r"would you rather", r"this or that"]),
    ("VRAI_FAUX",          [r"vrai ou faux", r"vrai\s*/\s*faux", r"true or false", r"\btrue\s*/?\s*false\b"]),
    ("OUI_NON",            [r"\boui ou non\b", r"\byes or no\b"]),
    ("LETTRES_MANQUANTES", [r"lettres manquantes", r"missing letters", r"compl[èe]te le mot", r"fill the word"]),
    ("MOT_MELANGE",        [r"mot m[ée]lang[ée]", r"lettres? dans le d[ée]sordre", r"scrambled word", r"unscramble"]),
    ("LETTRE_UNIQUE",      [r"une lettre", r"mot qui commence par", r"word starting with"]),
    ("TROUVER_INTRUS",     [r"intrus", r"odd one out", r"find the intruder"]),
    ("DILEMME_MORAL",      [r"dilemme", r"moral dilemma"]),
    ("CHAINE_DEFINITIONS", [r"d[ée]finition", r"3 indices", r"guess from clues", r"definition"]),
    ("DEVINE",             [r"\bdevine\b", r"\bguess\b", r"devinez", r"quel est"]),  # générique visuel -> en dernier
]


def infer_family(title: str | None, description: str | None = None) -> str | None:
    """
    Infère le code de famille. Le TITRE prime sur la description (qui contient souvent du
    texte promotionnel parasite) : on cherche d'abord dans le titre, puis en repli la description.
    """
    for text in (title, description):
        if not text or not text.strip():
            continue
        low = text.lower()
        for code, patterns in _FAMILY_PATTERNS:
            for pat in patterns:
                if re.search(pat, low):
                    return code
    return None


# ── Catégorie (inférence par mots-clés des seeds) ─────────────
def build_category_matcher(seeds_by_category: dict[str, dict[str, list[str]]]) -> dict[str, set[str]]:
    """Construit un index {catégorie: set(mots-clés normalisés)} à partir des seeds."""
    index: dict[str, set[str]] = {}
    for category, by_lang in seeds_by_category.items():
        toks: set[str] = set()
        for seeds in by_lang.values():
            for s in seeds:
                toks.update(_tokenize(s))
        # on retire les mots trop génériques au quiz
        toks -= {"quiz", "devine", "guess", "test", "blind", "general"}
        index[category] = toks
    return index


def infer_category(text: str, matcher: dict[str, set[str]], *, default: str = "QUIZ_GLOBAL") -> str:
    """Catégorie au meilleur recouvrement de tokens avec les seeds. Repli sur `default`."""
    tokens = set(_tokenize(text))
    best, best_score = default, 0
    for category, kws in matcher.items():
        score = len(tokens & kws)
        if score > best_score:
            best, best_score = category, score
    return best


# ── tokenisation simple ───────────────────────────────────────
_WORD_RE = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "") if len(w) > 2]


def label_record(
    rec: dict,
    *,
    category_hint: str | None,
    language_hint: str | None,
    matcher: dict[str, set[str]] | None = None,
) -> dict:
    """
    Ajoute les 3 labels à un enregistrement vidéo normalisé.
    `*_hint` = info connue au moment de la collecte (seed / chaîne).
    """
    title, desc = rec.get("title"), rec.get("description")
    language = detect_language(f"{title} {desc}", fallback=language_hint or "FR")
    if category_hint and taxonomy.is_valid_category(category_hint):
        category = category_hint
    elif matcher is not None:
        category = infer_category(f"{title} {desc}", matcher)
    else:
        category = "QUIZ_GLOBAL"
    family = infer_family(title, desc)
    return {
        **rec,
        "language": language,
        "category": category,
        "family": family,
        "family_group": taxonomy.family_group(family) if family else None,
    }
