"""
Collecteur Google Trends (pytrends) — sujets tendance par catégorie × langue.

Pour chaque catégorie et chaque langue, on récupère :
  - les requêtes associées montantes / top (related_queries) à partir des seeds
  - l'intérêt relatif dans le temps (interest_over_time) — signal de tendance

⚠️ Google Trends n'a pas d'API officielle : pytrends fait du reverse-engineering.
   Risque de rate-limiting (HTTP 429) -> on espace les requêtes et on dégrade proprement.
"""
from __future__ import annotations

import logging
import time

from .. import taxonomy

log = logging.getLogger(__name__)

try:
    from pytrends.request import TrendReq
except Exception:  # pragma: no cover - import optionnel
    TrendReq = None


class TrendsCollector:
    def __init__(self, *, sleep: float = 2.0):
        if TrendReq is None:
            raise RuntimeError("pytrends non installé. `pip install -r requirements.txt`")
        self.sleep = sleep  # délai anti rate-limit entre requêtes

    def _client(self, lang: str) -> "TrendReq":
        loc = taxonomy.LANGUAGE_LOCALE[lang]
        return TrendReq(hl=loc["trends_hl"], tz=0)

    def related_for_seed(self, seed: str, lang: str) -> dict:
        """Retourne les requêtes associées (top + rising) pour une seed donnée."""
        loc = taxonomy.LANGUAGE_LOCALE[lang]
        client = self._client(lang)
        out = {"seed": seed, "lang": lang, "top": [], "rising": []}
        try:
            client.build_payload([seed], timeframe="today 3-m", geo=loc["trends_geo"])
            rq = client.related_queries().get(seed) or {}
            if rq.get("top") is not None:
                out["top"] = rq["top"].to_dict("records")
            if rq.get("rising") is not None:
                out["rising"] = rq["rising"].to_dict("records")
        except Exception as e:  # rate-limit, payload vide, etc.
            log.warning("Trends related '%s' (%s) : %s", seed, lang, e)
        time.sleep(self.sleep)
        return out

    def collect(self, seeds_by_category: dict[str, dict[str, list[str]]]) -> list[dict]:
        """
        Parcourt {catégorie: {langue: [seeds]}} et agrège les sujets tendance.
        Retourne une liste de dicts étiquetés (catégorie, langue).
        """
        rows: list[dict] = []
        for category, by_lang in seeds_by_category.items():
            for lang, seeds in by_lang.items():
                for seed in seeds:
                    res = self.related_for_seed(seed, lang)
                    for kind in ("top", "rising"):
                        for rec in res[kind]:
                            rows.append({
                                "category": category,
                                "language": lang,
                                "seed": seed,
                                "trend_type": kind,         # top | rising
                                "query": rec.get("query"),
                                "value": rec.get("value"),
                            })
        return rows
