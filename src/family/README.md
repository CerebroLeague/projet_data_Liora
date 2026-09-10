# Classifieur de famille

Second modèle supervisé du projet : prédire la famille d'un quiz à partir de son titre
(classification multiclasses). Objectif concret : combler les ~40 % de vidéos dont la
famille n'a pas pu être déduite par règles.

Même démarche que le scoring : baseline (classe majoritaire) → régression logistique →
LightGBM, validation croisée, et une métrique adaptée au déséquilibre — le **macro-F1**
(l'accuracy seule est trompeuse ici, une classe écrase les autres).

## Lancer

```bash
python -m src.family.run_family
```

Produit le modèle (`models/family_classifier.joblib`), les métriques
(`reports/family_metrics.json`), la matrice de confusion
(`reports/figures/08_family_confusion.png`) et les familles comblées
(`data/processed/family_predictions.csv`).

## Résultat

Sur les données actuelles, LightGBM atteint ~0,90 d'accuracy et ~0,76 de macro-F1, là où la
baseline plafonne à 0,30 de macro-F1 malgré 0,84 d'accuracy — bonne illustration de
l'importance de la métrique face à un jeu déséquilibré.

## Limite

Les étiquettes d'entraînement viennent des règles d'étiquetage : le modèle apprend donc à
généraliser ces règles au-delà des titres qu'elles reconnaissaient. Les familles trop rares
(2–3 exemples) sont écartées de l'apprentissage.
