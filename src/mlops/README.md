# Tâche quotidienne & suivi

Chaque matin, on régénère le menu du jour : génération des idées, scoring, sélection et
publication dans `data/processed/suggestions.json` (que l'API et la démo lisent). Aucun
réentraînement ici, uniquement de l'inférence.

Un suivi simple accompagne chaque exécution : le score moyen des idées est enregistré, et
on signale une alerte s'il s'écarte trop de l'historique (dérive possible → envisager un
réentraînement).

## Lancer

```bash
python -m src.mlops.daily_pipeline                       # toutes les catégories
python -m src.mlops.daily_pipeline --category QUIZ_FOOT --n 5
```

## Automatisation

Le workflow [`.github/workflows/daily.yml`](../../.github/workflows/daily.yml) exécute cette
tâche tous les jours (cron). En Docker, le service `scheduler` s'en charge. Pour un horaire
précis, préférer un cron sur l'hôte.
