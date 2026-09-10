# Features & exploration

Transforme les données brutes en un dataset prêt pour l'entraînement, et produit les
figures d'exploration.

## Variable cible

Les vues brutes ne sont pas comparables d'une vidéo à l'autre. On construit donc un score
relatif : vues par jour, passées au logarithme, puis centrées-réduites *par niche*
(catégorie × langue). Un score de 0 correspond à une vidéo moyenne dans sa niche, positif
au-dessus.

## Éviter la fuite de données

La cible dérivant des vues, on exclut des features tout ce qui vient des vues (vues, likes,
commentaires). Le modèle ne voit que des signaux disponibles avant publication : titre,
durée, format, moment de publication, catégorie, famille, langue. Un garde-fou
(`assert_no_leakage`) échoue si une colonne interdite se glisse dans les features.

## Lancer

```bash
python -m src.features.run_features
```

Produit `data/processed/dataset.csv`, un profil (`profile.json`) et les figures dans
`reports/figures/`.
