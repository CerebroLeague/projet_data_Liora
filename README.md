# Quiz IA

Assistant d'idéation pour une chaîne YouTube de quiz vidéo. Le système propose chaque
matin un menu d'idées bilingues (français / anglais) et les classe par potentiel viral
estimé, à partir de ce qui fonctionne réellement chez les concurrents.

Projet de fin d'étude Data Science. Le détail méthodologique se trouve dans [`reports/`](reports/).

## Le problème

Produire une vidéo est automatisé ; trouver chaque jour une bonne idée ne l'est pas.
C'est ce goulot d'étranglement que le projet adresse : générer des idées pertinentes et
prédire lesquelles ont le plus de chances de marcher, avant même de les produire.

## Comment ça marche

Deux briques complémentaires :

- **Scoring viral** — un modèle (LightGBM) qui prédit un score de viralité à partir de
  signaux disponibles avant publication (titre, durée, format, catégorie…). Entraîné sur
  des métadonnées de vidéos concurrentes collectées via l'API YouTube.
- **Génération (RAG)** — on récupère les titres qui marchent, les tendances et les
  mots-clés d'une niche, on les donne en contexte à un LLM, et on lui fait produire des
  idées. Chaque idée est ensuite notée par le modèle de scoring, puis classée.

Le tout est exposé par une API et une démo Streamlit. Une tâche quotidienne régénère le
menu du matin automatiquement.

## Stack

Python 3.12 · scikit-learn / LightGBM · SHAP · TF-IDF + LLM (OpenAI / Mistral / Anthropic)
· FastAPI · Streamlit · Docker · GitHub Actions.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # renseigner les clés (YouTube, LLM)
```

## Utilisation

Le pipeline s'exécute dans cet ordre (chaque étape produit ce dont la suivante a besoin) :

```bash
python -m src.ingestion.run_ingestion      # 1. collecter les vidéos concurrentes
python -m src.features.run_features         # 2. dataset + figures
python -m src.scoring.run_scoring           # 3. entraîner le modèle de scoring
python -m src.generator.run_generator --category QUIZ_FOOT --language FR   # 4. générer
python -m src.mlops.daily_pipeline          # 5. générer le menu du jour
```

Servir l'application :

```bash
uvicorn src.api.app:app --reload            # API  -> http://127.0.0.1:8000/docs
streamlit run streamlit_app/app.py          # démo -> http://localhost:8501
```

Sans clé LLM, la génération bascule sur un mode de repli ; le scoring, lui, reste réel.

## Déploiement

Docker Compose (API + démo + tâche quotidienne) et déploiement continu par GitHub Actions.
Voir [docs/setup/deploiement_vps.md](docs/setup/deploiement_vps.md).

```bash
docker compose up -d
```

## Organisation du dépôt

```
src/            code source, un sous-dossier par étape (voir src/README.md)
streamlit_app/  démo
notebooks/      exploration et modélisation (vitrine des modules)
data/           données brutes et préparées
models/         modèle entraîné
reports/        rapports + figures
config/         paramètres de collecte (mots-clés, chaînes)
postman/        collection de test de l'API
docs/           consignes, schémas, guides de déploiement
```

Chaque dossier de `src/` a son propre README.

## Données

476 vidéos concurrentes collectées, réparties sur 7 chaînes thématiques (Foot, Basket,
K-Pop, Otaku, Global, DataWorld, Religion) en français et en anglais. Chaque vidéo est
étiquetée sur trois axes : **catégorie × langue × famille de quiz** (QCM, Vrai/Faux,
Devine, Rébus Emojis…).
