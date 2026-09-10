"""
Extraction de mots-clés agrégés par (catégorie × langue).

À partir des titres / tags / descriptions des vidéos concurrentes collectées, on calcule
les unigrammes et bigrammes les plus fréquents par axe. Sert à :
  - documenter les sujets porteurs par chaîne dans le rapport
  - alimenter le feature engineering du scoring viral (Sprint B)

Approche volontairement légère (Counter + stopwords FR/EN), sans dépendance NLP lourde.
"""
from __future__ import annotations

import re
from collections import Counter

_WORD_RE = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)

_STOP = {
    # FR
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à", "au", "aux",
    "en", "dans", "sur", "pour", "par", "avec", "ce", "cette", "ces", "qui", "que",
    "quoi", "est", "sont", "tu", "vous", "je", "il", "elle", "on", "ne", "pas", "plus",
    "son", "sa", "ses", "the", "quel", "quelle",
    # EN
    "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "with", "this", "that",
    "is", "are", "you", "your", "it", "its", "what", "who", "which", "from",
    # génériques quiz (bruit)
    "quiz", "test", "video", "vidéo", "shorts", "short",
    # bruit URL / réseaux (descriptions YouTube)
    "http", "https", "www", "com", "fr", "youtube", "youtu", "instagram",
    "tiktok", "abonne", "abonnez", "subscribe", "lien", "bit", "ly",
}


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "") if len(w) > 2 and w.lower() not in _STOP]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def extract_keywords(
    records: list[dict],
    *,
    top_k: int = 25,
) -> list[dict]:
    """
    Agrège les mots-clés par (category, language).
    `records` = vidéos étiquetées (champs: category, language, title, description, tags).
    Retourne des lignes {category, language, ngram, term, freq}.
    """
    buckets: dict[tuple[str, str], Counter] = {}
    for r in records:
        key = (r.get("category"), r.get("language"))
        if key[0] is None or key[1] is None:
            continue
        text = " ".join([
            r.get("title") or "",
            r.get("description") or "",
            " ".join(r.get("tags") or []),
        ])
        toks = _tokens(text)
        c = buckets.setdefault(key, Counter())
        c.update(toks)                       # unigrammes
        c.update(_ngrams(toks, 2))           # bigrammes

    rows: list[dict] = []
    for (category, language), counter in buckets.items():
        for term, freq in counter.most_common(top_k):
            rows.append({
                "category": category,
                "language": language,
                "ngram": 2 if " " in term else 1,
                "term": term,
                "freq": freq,
            })
    return rows
