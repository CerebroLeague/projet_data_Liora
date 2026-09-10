"""
Retriever du RAG — recherche filtrée par (catégorie × langue), TF-IDF pour le sémantique.

Choix : TF-IDF (scikit-learn) plutôt qu'embeddings neuronaux, car :
  - le corpus est petit et très spécialisé (niche quiz) → TF-IDF suffit et est pertinent ;
  - zéro dépendance lourde (pas de torch/faiss), fonctionne hors-ligne et gratuitement.
(Amélioration possible documentée : passer à des embeddings type sentence-transformers.)
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import corpus


class Retriever:
    def __init__(self, docs: list[dict] | None = None):
        self.docs = docs if docs is not None else corpus.load_documents()
        texts = [d["text"] for d in self.docs]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    # ── filtres ────────────────────────────────────────────────
    def _mask(self, *, category=None, language=None, source=None) -> list[int]:
        idx = []
        for i, d in enumerate(self.docs):
            if category and d["category"] != category:
                continue
            if language and d["language"] != language:
                continue
            if source and d["source"] != source:
                continue
            idx.append(i)
        return idx

    # ── récupération par succès (style) ────────────────────────
    def top_examples(self, category: str, language: str, k: int = 8) -> list[dict]:
        """Titres concurrents les plus vus dans la niche (grounding stylistique)."""
        idx = self._mask(category=category, language=language, source="style")
        ranked = sorted((self.docs[i] for i in idx),
                        key=lambda d: d["meta"].get("views", 0), reverse=True)
        return ranked[:k]

    def top_trends(self, category: str, language: str, k: int = 6) -> list[dict]:
        idx = self._mask(category=category, language=language, source="trend")
        ranked = sorted((self.docs[i] for i in idx),
                        key=lambda d: d["meta"].get("value", 0) or 0, reverse=True)
        return ranked[:k]

    def top_keywords(self, category: str, language: str, k: int = 10) -> list[dict]:
        idx = self._mask(category=category, language=language, source="keyword")
        ranked = sorted((self.docs[i] for i in idx),
                        key=lambda d: d["meta"].get("freq", 0) or 0, reverse=True)
        return ranked[:k]

    # ── recherche sémantique (thème libre, on-demand) ──────────
    def semantic_search(self, query: str, *, category=None, language=None, k: int = 8) -> list[dict]:
        if self.matrix is None or not query.strip():
            return []
        idx = self._mask(category=category, language=language)
        if not idx:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix[idx]).ravel()
        order = np.argsort(sims)[::-1][:k]
        return [self.docs[idx[j]] for j in order if sims[j] > 0]

    # ── contexte complet pour la génération ────────────────────
    def build_context(self, category: str, language: str, *, theme: str | None = None) -> dict:
        """Assemble le contexte RAG à injecter dans le prompt."""
        examples = (self.semantic_search(theme, category=category, language=language, k=8)
                    if theme else self.top_examples(category, language))
        return {
            "examples": [d["text"] for d in examples],
            "trends": [d["text"] for d in self.top_trends(category, language)],
            "keywords": [d["text"] for d in self.top_keywords(category, language)],
        }
