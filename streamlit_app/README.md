# Démo

Petite application Streamlit qui sert de vitrine au projet. Le modèle est chargé une fois,
jamais réentraîné pendant la démo.

Quatre onglets : le menu du jour, la génération à la demande, l'explication d'un score
(SHAP) et les figures d'exploration.

## Lancer

```bash
streamlit run streamlit_app/app.py     # http://localhost:8501
```

Avec une clé LLM dans `.env`, la génération est réelle ; sinon un mode repli prend le
relais et le scoring reste réel.

Pour que tout s'affiche, avoir lancé au préalable `src.features.run_features` (figures) et
`src.scoring.run_scoring` (métriques et SHAP).
