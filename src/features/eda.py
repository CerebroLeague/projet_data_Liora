"""
Exploration (EDA) & DataViz — génère les figures du Rendu 1.

Chaque fonction produit une figure pertinente (≥5 exigées par la consigne) et la sauvegarde
dans reports/figures/. Les figures sont pensées pour être commentées (avis métier) et
validées par une stat dans le rapport.

Backend non-interactif (Agg) pour fonctionner en script / CI.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid")
ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def profile(df: pd.DataFrame) -> dict:
    """Statistiques descriptives de base (volumétrie, NA, couverture des axes)."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "na_rate": (df.isna().mean().round(3)).to_dict(),
        "by_category": df["category"].value_counts().to_dict() if "category" in df else {},
        "by_language": df["language"].value_counts().to_dict() if "language" in df else {},
        "by_family": df["family"].value_counts(dropna=False).to_dict() if "family" in df else {},
    }


# ── Figures ───────────────────────────────────────────────────
def fig_virality_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["virality_score"].dropna(), kde=True, ax=ax, color="#7C3AED")
    ax.set(title="Distribution du score de viralité (z-score par niche)",
           xlabel="virality_score", ylabel="nb de vidéos")
    return _save(fig, "01_virality_distribution.png")


def fig_volume_by_category_lang(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.countplot(data=df, x="category", hue="language", ax=ax)
    ax.set(title="Volume de vidéos collectées par catégorie et langue",
           xlabel="catégorie", ylabel="nb de vidéos")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "02_volume_by_category_language.png")


def fig_views_per_day_by_category(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = df.copy()
    d["log_vpd"] = np.log1p(d["views_per_day"])
    sns.boxplot(data=d, x="category", y="log_vpd", ax=ax)
    ax.set(title="Vitesse d'accumulation des vues par catégorie (échelle log)",
           xlabel="catégorie", ylabel="log(1 + vues/jour)")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "03_views_per_day_by_category.png")


def fig_family_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = df.copy()
    d["family"] = d["family"].fillna("UNKNOWN").astype(str)  # seaborn refuse les NaN
    order = d["family"].value_counts().index
    sns.countplot(data=d, y="family", order=order, ax=ax, color="#2DD4BF")
    ax.set(title="Répartition des familles de quiz (inférées)",
           xlabel="nb de vidéos", ylabel="famille")
    return _save(fig, "04_family_distribution.png")


def fig_duration_vs_virality(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = df.dropna(subset=["duration_seconds", "virality_score"])
    sns.scatterplot(data=d, x="duration_seconds", y="virality_score",
                    hue="language", alpha=0.6, ax=ax)
    ax.set(title="Durée de la vidéo vs viralité",
           xlabel="durée (s)", ylabel="virality_score")
    return _save(fig, "05_duration_vs_virality.png")


def fig_fr_vs_en(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    d = df.copy()
    d["log_vpd"] = np.log1p(d["views_per_day"])
    sns.boxplot(data=d, x="language", y="log_vpd", ax=ax, palette=["#5b8def", "#f06595"])
    ax.set(title="Vues par jour : FR vs EN (échelle log)",
           xlabel="langue", ylabel="log(1 + vues/jour)")
    return _save(fig, "09_fr_vs_en.png")


def fig_correlation_heatmap(df: pd.DataFrame) -> Path:
    num = df.select_dtypes(include="number")
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, annot=False)
    ax.set(title="Corrélations entre variables numériques")
    return _save(fig, "06_correlation_heatmap.png")


def generate_all(df: pd.DataFrame) -> list[Path]:
    """Génère toutes les figures disponibles selon les colonnes présentes."""
    figs = []
    if "virality_score" in df:
        figs.append(fig_virality_distribution(df))
        if "category" in df:
            figs.append(fig_views_per_day_by_category(df))
        if {"duration_seconds"} <= set(df.columns):
            figs.append(fig_duration_vs_virality(df))
    if {"category", "language"} <= set(df.columns):
        figs.append(fig_volume_by_category_lang(df))
    if {"language", "views_per_day"} <= set(df.columns):
        figs.append(fig_fr_vs_en(df))
    if "family" in df:
        figs.append(fig_family_distribution(df))
    figs.append(fig_correlation_heatmap(df))
    return figs
