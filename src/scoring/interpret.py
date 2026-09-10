"""
Interprétabilité du scoring viral via SHAP (TreeExplainer pour LightGBM).

Produit :
  - un summary plot (importance + sens des features) -> reports/figures/
  - le classement des features par importance moyenne |SHAP|
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures"


def explain(model, X: pd.DataFrame) -> pd.DataFrame:
    """Calcule les valeurs SHAP, sauvegarde le summary plot, renvoie l'importance triée."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title("SHAP — facteurs de viralité prédite")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    # importance moyenne |SHAP|
    importance = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance
