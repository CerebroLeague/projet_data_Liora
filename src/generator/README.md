# Génération (RAG)

Génère des idées de quiz ancrées sur des données réelles, sans réentraîner de modèle.

Le principe : pour une niche donnée, on récupère les titres qui marchent, les tendances du
moment et les mots-clés porteurs, on les fournit en contexte à un LLM, et on lui fait
produire des idées. Chaque idée est ensuite notée par le modèle de scoring et classée.

La recherche de contexte utilise TF-IDF plutôt que des embeddings lourds : le corpus est
petit et spécialisé, ça suffit et ça tourne hors-ligne. Seule la génération finale passe
par une API.

## Configurer le LLM

Dans `.env` :

```
LLM_PROVIDER=openai        # ou mistral, anthropic
LLM_API_KEY=...
LLM_MODEL=                  # vide = modèle par défaut du fournisseur
```

## Lancer

```bash
# sans clé : mode repli (idées issues du contexte, scoring réel)
python -m src.generator.run_generator --category QUIZ_FOOT --language FR --mock

# avec clé
python -m src.generator.run_generator --category QUIZ_KPOP --language EN --n 5

# menu du jour FR + EN
python -m src.generator.run_generator --menu --category QUIZ_FOOT --save
```
