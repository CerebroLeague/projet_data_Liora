# Interface de test

Petite UI HTML autonome pour visualiser les résultats de l'API pendant les tests. Elle est
**indépendante de l'API** (elle l'appelle via HTTP) et n'a rien à voir avec les frontends de
production.

## Lancer

Dans deux terminaux, depuis la racine du dépôt :

```bash
# 1. l'API
uvicorn src.api.app:app --port 8000

# 2. l'interface (origine déjà autorisée par le CORS de l'API)
python -m http.server 3003 --directory web
```

Puis ouvrir http://localhost:3003.

L'UI cible l'API sur `http://localhost:8000`. Si l'API tourne ailleurs, ajuster la constante
`API` en haut du `<script>` dans `index.html`.
