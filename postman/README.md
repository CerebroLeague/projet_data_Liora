# Tester l'API avec Postman

## Démarrer l'API

```bash
uvicorn src.api.app:app --port 8000
```

Le modèle et le menu du jour doivent exister. Sinon :

```bash
python -m src.scoring.run_scoring --no-grid --no-shap
python -m src.mlops.daily_pipeline
```

## Importer la collection

Dans Postman : Import → `postman/quiz-ia.postman_collection.json`. La variable `base_url`
pointe sur `http://127.0.0.1:8000`.

La collection contient les quatre requêtes : `/health`, `/suggestions/today`, `/generate`
et `/score`. On peut aussi tout tester depuis la doc interactive sur `/docs`.

## Génération réelle

Sans clé LLM, `/generate` renvoie des idées en mode repli (`llm_used: false`). Pour de
vraies idées, renseigner `LLM_PROVIDER` et `LLM_API_KEY` dans `.env`, puis relancer l'API.

Catégories : `QUIZ_FOOT`, `QUIZ_BASKET`, `QUIZ_KPOP`, `QUIZ_OTAKU`, `QUIZ_GLOBAL`,
`QUIZ_DATAWORLD`, `QUIZ_RELIGION`. Langues : `FR`, `EN`.
