"""
Client LLM provider-agnostique (via API HTTP) — étape de génération du RAG.

Fournisseurs supportés (choisis via `LLM_PROVIDER` dans .env) :
  - mistral   : https://api.mistral.ai/v1/chat/completions   (défaut : mistral-large-latest)
  - openai    : https://api.openai.com/v1/chat/completions    (défaut : gpt-4o-mini)
  - anthropic : https://api.anthropic.com/v1/messages         (défaut : claude-sonnet-4-6)

On utilise `requests` directement (pas de SDK) pour rester léger et uniforme.
"""
from __future__ import annotations

import os

import requests

_DEFAULT_MODEL = {
    "mistral": "mistral-large-latest",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
}
_ENDPOINT = {
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
}


class LLMClient:
    def __init__(self, provider: str | None = None, api_key: str | None = None,
                 model: str | None = None, *, timeout: int = 60):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "mistral").lower()
        if self.provider not in _ENDPOINT:
            raise ValueError(f"LLM_PROVIDER inconnu: {self.provider} (mistral|openai|anthropic)")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY manquante. Renseigner .env (cf. .env.example).")
        self.model = model or os.getenv("LLM_MODEL") or _DEFAULT_MODEL[self.provider]
        self.timeout = timeout

    def chat(self, system: str, user: str, *, temperature: float = 0.8, max_tokens: int = 1500) -> str:
        """Envoie un prompt (system + user) et retourne le texte de la réponse."""
        if self.provider == "anthropic":
            return self._anthropic(system, user, temperature, max_tokens)
        return self._openai_compatible(system, user, temperature, max_tokens)

    # ── OpenAI & Mistral (format identique) ────────────────────
    def _openai_compatible(self, system, user, temperature, max_tokens) -> str:
        r = requests.post(
            _ENDPOINT[self.provider],
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # ── Anthropic (format messages) ────────────────────────────
    def _anthropic(self, system, user, temperature, max_tokens) -> str:
        r = requests.post(
            _ENDPOINT["anthropic"],
            headers={"x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]
