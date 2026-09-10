"""
Monitoring de dérive (léger) sur l'historique des runs quotidiens.

Idée : suivre dans le temps le **score moyen** des idées générées. Une variation brutale
signale que quelque chose a changé (tendances, corpus, comportement du LLM) → envisager
une ré-évaluation / ré-entraînement du scoring.

Règle simple et transparente : z-score du dernier run vs la moyenne glissante des runs
précédents ; alerte si |z| > seuil.
"""
from __future__ import annotations

import json
from pathlib import Path


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def check(history_path: Path, *, z_threshold: float = 2.0, min_runs: int = 4) -> list[str]:
    """Retourne une liste de messages d'alerte (vide si RAS)."""
    hist = _load_history(history_path)
    if len(hist) < min_runs:
        return []  # pas assez d'historique pour statuer

    scores = [h.get("mean_score", 0.0) for h in hist]
    *past, last = scores
    mean = sum(past) / len(past)
    var = sum((s - mean) ** 2 for s in past) / len(past)
    std = var ** 0.5
    flags: list[str] = []
    if std > 0:
        z = (last - mean) / std
        if abs(z) > z_threshold:
            flags.append(
                f"score moyen du dernier run = {last:.3f} (z={z:+.2f} vs historique {mean:.3f}) "
                f"→ dérive potentielle, envisager un ré-entraînement du scoring."
            )
    return flags
