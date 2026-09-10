# Obtenir une clé YouTube Data API v3 (gratuit, ~5 min)

Guide express pour générer la clé attendue dans `.env` (`YOUTUBE_API_KEY`).

---

## Étapes

### 1. Ouvrir la console Google Cloud
👉 https://console.cloud.google.com/

Connecte-toi avec ton compte Google (`nagorfonkoua@gmail.com`).

### 2. Créer un projet
- En haut à gauche, clique sur le **sélecteur de projet** → **Nouveau projet**.
- Nom : `heroquiz-ia` (ou ce que tu veux) → **Créer**.
- Attends quelques secondes, puis **sélectionne ce projet** (toujours en haut).

### 3. Activer l'API YouTube Data v3
- Menu (☰) → **APIs et services** → **Bibliothèque**.
- Recherche : `YouTube Data API v3`.
- Clique dessus → bouton **Activer**.

### 4. Créer la clé d'API
- Menu (☰) → **APIs et services** → **Identifiants** (Credentials).
- Bouton **+ Créer des identifiants** → **Clé API**.
- Une clé s'affiche (format `AIza...`) → **copie-la**.

### 5. (Recommandé) Restreindre la clé
Toujours dans **Identifiants**, clique sur ta clé pour l'éditer :
- **Restrictions relatives aux API** → *Restreindre la clé* → coche **YouTube Data API v3** → Enregistrer.

> Cela évite qu'une clé qui fuite serve à autre chose. Pas de restriction d'IP nécessaire
> pour un usage local en développement.

### 6. Coller la clé dans le projet
Dans le dossier du projet :

```bash
cp .env.example .env
```

Puis édite `.env` :

```
YOUTUBE_API_KEY=AIza...ta_cle...
```

> ⚠️ Le fichier `.env` est déjà dans `.gitignore` — ta clé ne sera jamais commitée.

---

## Vérifier que ça marche

```bash
. .venv/bin/activate
python -m src.ingestion.run_ingestion --categories QUIZ_FOOT --languages FR --max-per-query 5 --skip-trends
```

Si tu vois des lignes `search '...' → N vidéos` puis l'écriture de
`data/raw/competitors/videos.csv`, c'est bon. ✅

---

## Bon à savoir sur le quota

- **Gratuit**, sans carte bancaire.
- **10 000 unités/jour** (remis à zéro chaque jour, heure du Pacifique).
- Coût des opérations : `search.list` = **100 unités**, `videos.list` = **1 unité**.
- 👉 Commence petit (`--max-per-query 5`, une catégorie) pour valider, puis élargis.
- Si tu vois `Quota dépassé` : ce n'est pas grave, le script sauvegarde ce qui a été récupéré ;
  on relance le lendemain ou on demande une augmentation de quota.

---

## En cas de souci

| Message | Cause probable | Solution |
|---|---|---|
| `YOUTUBE_API_KEY manquante` | `.env` absent ou vide | refaire l'étape 6 |
| `API ... not enabled` | API pas activée | refaire l'étape 3 |
| `quota` / 403 | quota du jour épuisé | attendre le reset ou réduire `--max-per-query` |
