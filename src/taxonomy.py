"""
Taxonomie de référence du projet — miroir Python du code CerebroLeague.

Trois axes de classement pour toute donnée collectée :
  - FAMILLE de quiz   (FormatFamilyCode, groupée par grande famille)
  - CATÉGORIE         (QuizCategory, 20 types canoniques)
  - LANGUE            (FR / EN)

Sources canoniques :
  - src/types/format.ts   (familles)
  - src/types/content.ts  (catégories)
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Axe 1 — Familles de quiz
# ─────────────────────────────────────────────────────────────

# code -> (libellé, grande famille). Les familles actives en production d'abord.
FAMILIES: dict[str, tuple[str, str]] = {
    # ── Choix ──
    "MCQ":                ("QCM", "Choix"),
    "VRAI_FAUX":          ("Vrai / Faux", "Choix"),
    "OUI_NON":            ("Oui / Non", "Choix"),
    "TROUVER_INTRUS":     ("Trouver l'Intrus", "Choix"),
    "DILEMME_MORAL":      ("Dilemme moral", "Choix"),
    # ── Devine le Mot ──
    "CHAINE_DEFINITIONS": ("Chaîne de Définitions", "Devine le Mot"),
    "LETTRES_MANQUANTES": ("Lettres Manquantes", "Devine le Mot"),
    "MOT_MELANGE":        ("Mot Mélangé", "Devine le Mot"),
    "REBUS_EMOJIS":       ("Rébus Emojis", "Devine le Mot"),
    "LETTRE_UNIQUE":      ("Lettre Unique", "Devine le Mot"),
    # ── Visuel ──
    "DEVINE":             ("Devine", "Visuel"),
    # ── Sport ──
    "COMPO_DRAPEAUX":     ("Compo Drapeaux", "Sport"),
    # ── Personnalité ──
    "TU_PREFERES":        ("Tu Préfères", "Personnalité"),
}

# Familles actives en production (cf. screenshots de l'app)
ACTIVE_FAMILIES: list[str] = [
    "MCQ", "VRAI_FAUX", "OUI_NON",
    "LETTRES_MANQUANTES", "MOT_MELANGE", "REBUS_EMOJIS", "LETTRE_UNIQUE",
    "DEVINE", "COMPO_DRAPEAUX", "TU_PREFERES",
]

FAMILY_CODES: list[str] = list(FAMILIES.keys())


def family_label(code: str) -> str:
    return FAMILIES.get(code, (code, ""))[0]


def family_group(code: str) -> str:
    return FAMILIES.get(code, ("", "Autre"))[1]


# ─────────────────────────────────────────────────────────────
# Axe 2 — Catégories = les chaînes réellement exploitées
# ─────────────────────────────────────────────────────────────
# IMPORTANT : ce ne sont PAS les 20 QuizCategory génériques de CerebroLeague.
# Ce sont les catégories de la capture = une chaîne par catégorie, déclinée par langue
# (=> nb de chaînes = len(CATEGORIES) × len(LANGUAGES)).

CATEGORIES: list[str] = [
    "QUIZ_FOOT", "QUIZ_BASKET", "QUIZ_KPOP", "QUIZ_OTAKU",
    "QUIZ_GLOBAL", "QUIZ_DATAWORLD", "QUIZ_RELIGION",
]

# Libellés d'affichage (pour rapports / UI)
CATEGORY_LABELS: dict[str, str] = {
    "QUIZ_FOOT":      "Quiz Foot",
    "QUIZ_BASKET":    "Quiz Basket",
    "QUIZ_KPOP":      "Quiz K-Pop",
    "QUIZ_OTAKU":     "Quiz Otaku",
    "QUIZ_GLOBAL":    "Quiz Global",
    "QUIZ_DATAWORLD": "Quiz DataWorld",
    "QUIZ_RELIGION":  "Quiz Religion",
}

# Thème dominant (utile pour le rapport / regroupements)
CATEGORY_THEME: dict[str, str] = {
    "QUIZ_FOOT":      "Football",
    "QUIZ_BASKET":    "Basket",
    "QUIZ_KPOP":      "K-Pop / Musique",
    "QUIZ_OTAKU":     "Anime / Manga",
    "QUIZ_GLOBAL":    "Culture générale",
    "QUIZ_DATAWORLD": "Culture générale / Data",
    "QUIZ_RELIGION":  "Religion / Culture",
}

# ─────────────────────────────────────────────────────────────
# Axe 3 — Langues
# ─────────────────────────────────────────────────────────────

LANGUAGES: list[str] = ["FR", "EN"]

# Paramètres régionaux YouTube / Trends par langue
LANGUAGE_LOCALE: dict[str, dict[str, str]] = {
    "FR": {"region_code": "FR", "relevance_language": "fr", "trends_geo": "FR", "trends_hl": "fr-FR"},
    "EN": {"region_code": "US", "relevance_language": "en", "trends_geo": "US", "trends_hl": "en-US"},
}


def is_valid_category(cat: str) -> bool:
    return cat in CATEGORIES


def is_valid_language(lang: str) -> bool:
    return lang in LANGUAGES
