"""
Collecteur YouTube Data API v3 — métadonnées de vidéos concurrentes.

Deux modes de découverte :
  1. recherche par mots-clés (seeds par catégorie × langue)   -> search_by_query()
  2. vidéos d'une chaîne connue (playlist "uploads")           -> videos_of_channel()

Coût quota (quota par défaut = 10 000 unités/jour) :
  - search.list           : 100 unités / appel  (≤ 50 résultats)
  - videos.list           : 1 unité / appel     (≤ 50 ids)
  - channels.list         : 1 unité / appel
  - playlistItems.list    : 1 unité / appel     (≤ 50 ids)
=> La recherche est l'opération coûteuse : on la garde sous contrôle (max_results, nb de seeds).
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Iterator

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

log = logging.getLogger(__name__)


class QuotaExceeded(RuntimeError):
    """Levée quand l'API renvoie une erreur de quota dépassé."""


class YouTubeCollector:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "YOUTUBE_API_KEY manquante. Copier .env.example en .env et renseigner la clé."
            )
        self.yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        self.quota_used = 0  # estimation locale du quota consommé

    # ── helpers ────────────────────────────────────────────────
    def _spend(self, units: int) -> None:
        self.quota_used += units

    @staticmethod
    def _is_quota_error(e: HttpError) -> bool:
        return e.resp.status == 403 and b"quota" in e.content.lower()

    # ── 1. recherche par mots-clés ─────────────────────────────
    def search_by_query(
        self,
        query: str,
        *,
        region_code: str = "FR",
        relevance_language: str = "fr",
        max_results: int = 25,
        published_after: str | None = None,
    ) -> list[str]:
        """Retourne une liste d'IDs de vidéos correspondant à la requête."""
        ids: list[str] = []
        page_token: str | None = None
        try:
            while len(ids) < max_results:
                req = self.yt.search().list(
                    q=query,
                    part="id",
                    type="video",
                    maxResults=min(50, max_results - len(ids)),
                    regionCode=region_code,
                    relevanceLanguage=relevance_language,
                    order="relevance",
                    pageToken=page_token,
                    publishedAfter=published_after,
                )
                resp = req.execute()
                self._spend(100)
                ids += [it["id"]["videoId"] for it in resp.get("items", []) if it["id"].get("videoId")]
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as e:
            if self._is_quota_error(e):
                raise QuotaExceeded(f"Quota dépassé (utilisé ≈ {self.quota_used}).") from e
            log.warning("Erreur search '%s' : %s", query, e)
        return ids

    # ── 2. vidéos d'une chaîne ─────────────────────────────────
    def resolve_channel_id(self, *, handle: str | None = None, channel_id: str | None = None) -> str | None:
        """Résout un handle (@nom) ou renvoie l'ID tel quel."""
        if channel_id:
            return channel_id
        if not handle:
            return None
        try:
            resp = self.yt.channels().list(part="id", forHandle=handle.lstrip("@")).execute()
            self._spend(1)
            items = resp.get("items", [])
            return items[0]["id"] if items else None
        except HttpError as e:
            log.warning("Résolution handle '%s' échouée : %s", handle, e)
            return None

    def videos_of_channel(self, channel_id: str, *, max_results: int = 50) -> list[str]:
        """Liste les IDs des dernières vidéos uploadées par une chaîne (via playlist uploads)."""
        try:
            ch = self.yt.channels().list(part="contentDetails", id=channel_id).execute()
            self._spend(1)
            items = ch.get("items", [])
            if not items:
                return []
            uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        except HttpError as e:
            log.warning("channels.list '%s' : %s", channel_id, e)
            return []

        ids: list[str] = []
        page_token: str | None = None
        try:
            while len(ids) < max_results:
                resp = self.yt.playlistItems().list(
                    part="contentDetails",
                    playlistId=uploads,
                    maxResults=min(50, max_results - len(ids)),
                    pageToken=page_token,
                ).execute()
                self._spend(1)
                ids += [it["contentDetails"]["videoId"] for it in resp.get("items", [])]
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as e:
            log.warning("playlistItems.list '%s' : %s", channel_id, e)
        return ids

    # ── détails des vidéos (batch) ─────────────────────────────
    def video_details(self, video_ids: Iterable[str]) -> Iterator[dict]:
        """Récupère snippet + statistics + contentDetails par lots de 50 ids."""
        ids = list(dict.fromkeys(video_ids))  # dédoublonnage en gardant l'ordre
        for i in range(0, len(ids), 50):
            batch = ids[i : i + 50]
            try:
                resp = self.yt.videos().list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(batch),
                ).execute()
                self._spend(1)
            except HttpError as e:
                if self._is_quota_error(e):
                    raise QuotaExceeded(f"Quota dépassé (utilisé ≈ {self.quota_used}).") from e
                log.warning("videos.list batch : %s", e)
                continue
            for it in resp.get("items", []):
                yield self._normalize(it)

    @staticmethod
    def _normalize(item: dict) -> dict:
        """Aplatit la réponse API en un enregistrement tabulaire propre."""
        sn = item.get("snippet", {})
        st = item.get("statistics", {})
        cd = item.get("contentDetails", {})
        return {
            "video_id": item.get("id"),
            "title": sn.get("title"),
            "description": sn.get("description"),
            "channel_id": sn.get("channelId"),
            "channel_title": sn.get("channelTitle"),
            "published_at": sn.get("publishedAt"),
            "tags": sn.get("tags", []),
            "category_id_yt": sn.get("categoryId"),
            "default_language": sn.get("defaultLanguage") or sn.get("defaultAudioLanguage"),
            "duration_iso": cd.get("duration"),
            "view_count": int(st["viewCount"]) if "viewCount" in st else None,
            "like_count": int(st["likeCount"]) if "likeCount" in st else None,
            "comment_count": int(st["commentCount"]) if "commentCount" in st else None,
            "thumbnail": (sn.get("thumbnails", {}).get("high", {}) or {}).get("url"),
        }
