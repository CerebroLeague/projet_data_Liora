# Notebooks

Trois notebooks qui déroulent la démarche pas à pas. Ils s'appuient sur les modules de
`src/` (pas de code dupliqué) et servent à visualiser les résultats.

- `01_exploration.ipynb` — exploration des données, figures, tests statistiques
- `02_modelisation.ipynb` — baselines, LightGBM, optimisation, SHAP
- `03_generation_rag.ipynb` — démonstration du pipeline de génération

## Lancer

```bash
jupyter notebook
# ou tout ré-exécuter :
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Prérequis : données collectées et dataset généré (voir le README racine). Le notebook 03
utilise le vrai LLM si une clé est présente, sinon le mode repli.
