# API

API d'inférence, point d'intégration du système dans une application. Les modèles sont
chargés une fois au démarrage — rien n'est réentraîné à la volée.

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | état du service |
| GET | `/suggestions/today` | menu du jour pré-calculé |
| POST | `/generate` | générer des idées pour une niche |
| POST | `/score` | noter des idées fournies |

## Lancer

```bash
uvicorn src.api.app:app --reload
```

Documentation interactive sur http://127.0.0.1:8000/docs. Une collection Postman est
fournie dans [`postman/`](../../postman/).

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"category":"QUIZ_KPOP","language":"EN","n":5}'
```

Le champ `llm_used` indique si le vrai LLM a répondu ou si le mode repli a été utilisé.
