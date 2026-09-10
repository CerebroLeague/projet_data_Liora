# Code source

Le code est découpé par étape du pipeline. Chaque sous-dossier fait une chose et a son
propre README.

| Dossier | Rôle |
|---|---|
| [`ingestion/`](ingestion/) | collecte des vidéos concurrentes (YouTube, Trends) et étiquetage |
| [`features/`](features/) | variable cible, préparation des features, figures d'exploration |
| [`scoring/`](scoring/) | modèle de prédiction de viralité |
| [`family/`](family/) | classifieur de famille de quiz (comble les familles manquantes) |
| [`generator/`](generator/) | génération d'idées par RAG |
| [`api/`](api/) | API FastAPI |
| [`mlops/`](mlops/) | tâche quotidienne et suivi de dérive |

`taxonomy.py` est partagé par tous : il définit les familles de quiz, les 7 catégories
et les langues.

On lance les modules comme des packages, depuis la racine du dépôt :

```bash
python -m src.ingestion.run_ingestion --help
```
