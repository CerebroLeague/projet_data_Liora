# Déploiement sur VPS (Docker Compose)

Trois services partagent une seule image :

| Service | Rôle | Port hôte (défaut) |
|---|---|---|
| `api` | API d'inférence (FastAPI) | `API_PORT` = **8000** |
| `streamlit` | démo PoC | `STREAMLIT_PORT` = **8501** |
| `scheduler` | boucle quotidienne (menu du jour) | — |

> **Ports & conflits.** 8000 et 8501 sont libres sur le VPS (déjà pris : 3000, 3001, 3003, 3005, 5432).
> Ils sont **configurables** via `.env` (`API_PORT`, `STREAMLIT_PORT`) sans toucher au `docker-compose.yml` —
> ex. `API_PORT=3007 STREAMLIT_PORT=3008`.

## 1. Pré-requis sur le VPS
- Docker + Docker Compose (`docker compose version`).
- Le dépôt cloné.
- Un fichier `.env` (copié depuis `.env.example`) avec au moins `LLM_PROVIDER` + `LLM_API_KEY`
  (sinon la génération tourne en *mock*).

## 2. Bootstrap (une fois) — modèle & données
L'image ne contient **pas** les données ni le modèle (montés via volumes). Il faut donc,
la première fois, produire le modèle et un premier menu :

```bash
# construire l'image
docker compose build

# (a) si data/processed/dataset.csv n'existe pas encore : collecte + features
docker compose run --rm api python -m src.ingestion.run_ingestion --max-per-query 15
docker compose run --rm api python -m src.features.run_features

# (b) entraîner le modèle de scoring (crée models/scoring_viral.joblib)
docker compose run --rm api python -m src.scoring.run_scoring

# (c) premier menu du jour
docker compose run --rm api python -m src.mlops.daily_pipeline
```

> Les volumes `./data`, `./models`, `./reports` persistent ces artefacts sur le VPS.

## 3. Lancer
```bash
docker compose up -d          # démarre api + streamlit + scheduler
docker compose ps             # état
docker compose logs -f api    # logs
```

- API : `http://<IP_VPS>:8000` (docs : `/docs`)
- Démo : `http://<IP_VPS>:8501`

## 4. Mettre à jour
```bash
git pull
docker compose build
docker compose up -d
```

## 5. Arrêter
```bash
docker compose down
```

## Dépannage
- **`address already in use` sur :8000 / :8501** : un autre process (ou un `uvicorn`/`streamlit`
  lancé à la main) occupe le port. L'arrêter, puis `docker compose up -d --force-recreate`.
- **Conteneur « Up » mais sans port mappé** : recréer proprement — `docker compose down` puis
  `docker compose up -d --force-recreate`.
- **`model_loaded: false` / erreur au démarrage** : le modèle n'est pas monté. Refaire l'étape 2
  (bootstrap) pour générer `models/scoring_viral.joblib`.

## Déploiement automatique (CI/CD — GitHub Actions)

Le workflow [`.github/workflows/deploy.yml`](../../.github/workflows/deploy.yml) déploie sur le VPS
à chaque `push` sur `main` (ou manuellement) : rsync du code → `.env` depuis les secrets →
`docker compose build` → **bootstrap** (entraîne le modèle au 1ᵉʳ déploiement) → `up -d` → vérif.

> `data/`, `models/` et `reports/` sont **exclus du rsync** : les artefacts générés sur le serveur
> (modèle, menu du jour, historique) sont **préservés** entre deux déploiements.

### Secrets GitHub à configurer (Settings → Secrets → Actions)
| Secret | Rôle |
|---|---|
| `SERVER_HOST` | IP / domaine du VPS |
| `SERVER_USER` | utilisateur SSH |
| `SERVER_SSH_KEY` | clé privée SSH (accès au VPS) |
| `YOUTUBE_API_KEY` | collecte (bootstrap) |
| `LLM_PROVIDER` · `LLM_API_KEY` · `LLM_MODEL` | génération RAG |

### Pré-requis serveur
Docker + Docker Compose installés, et le dossier `DEPLOY_DIR` (`/opt/quiz-ia`) accessible à l'utilisateur SSH.

## Notes
- **Secrets** : injectés via `env_file: .env` (jamais dans l'image ; `.env` est dans `.dockerignore` et `.gitignore`).
- **Planification** : le service `scheduler` régénère le menu toutes les 24 h. Pour un horaire précis
  (ex. 06:00), préférer un cron hôte : `0 6 * * * cd /chemin && docker compose run --rm api python -m src.mlops.daily_pipeline`.
- **Reverse-proxy** : en production, placer un Nginx/Caddy devant (HTTPS) et ne pas exposer les ports en clair.
- **Ressources** : image basée sur `python:3.12-slim` + `libgomp1` (requis par LightGBM).
