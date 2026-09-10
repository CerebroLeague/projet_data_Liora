# Ingestion

Collecte des métadonnées de vidéos concurrentes et étiquetage sur trois axes :
catégorie, langue et famille de quiz.

La découverte se fait de deux manières : par mots-clés (définis dans
`config/categories_keywords.yaml`) et par chaînes connues (`config/channels.yaml`). On
récupère aussi les sujets tendance (Google Trends) et on agrège les mots-clés porteurs
par niche.

Les vidéos concurrentes n'étant pas étiquetées, la catégorie et la famille sont déduites
(règles sur le titre, recoupement de mots-clés). La famille reste indéterminée dans une
partie des cas — c'est une limite assumée, documentée dans le rapport.

## Lancer

Il faut une clé YouTube Data API v3 dans `.env` (voir
[../../docs/setup/cle_api_youtube.md](../../docs/setup/cle_api_youtube.md)).

```bash
# vérifier la config sans appeler l'API
python -m src.ingestion.run_ingestion --dry-run

# collecte ciblée (économe en quota)
python -m src.ingestion.run_ingestion --categories QUIZ_FOOT --languages FR --max-per-query 10

# collecte complète
python -m src.ingestion.run_ingestion
```

Une fois des données collectées, on peut extraire automatiquement les chaînes concurrentes
les plus présentes :

```bash
python -m src.ingestion.discover_channels --top 3 --write
```

## À savoir

Une recherche coûte 100 unités de quota (10 000 par jour). Commencer petit puis élargir.
Résultats écrits dans `data/raw/` (vidéos, tendances, mots-clés).
