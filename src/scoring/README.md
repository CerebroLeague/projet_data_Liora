# Scoring viral

Prédit le score de viralité d'une idée avant publication. C'est un problème de régression :
on cherche surtout à bien *ordonner* les idées, donc la métrique principale est la
corrélation de Spearman (complétée par MAE et RMSE).

On procède du simple au complexe : une baseline triviale (moyenne), une régression
linéaire, puis LightGBM optimisé par validation croisée. LightGBM est retenu — il dépasse
nettement les baselines. L'importance des variables est analysée avec SHAP.

## Lancer

```bash
python -m src.scoring.run_scoring              # complet
python -m src.scoring.run_scoring --no-grid --no-shap   # rapide
```

Produit le modèle (`models/scoring_viral.joblib`), la liste des colonnes attendues, les
métriques (`reports/scoring_metrics.json`) et le graphe SHAP.

Sur le corpus actuel (476 vidéos), Spearman est autour de 0,48 : l'approche est validée,
la performance reste à améliorer avec plus de données. Détails dans
[`reports/rendu_2.md`](../../reports/rendu_2.md).
