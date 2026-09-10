"""
Classifieur de famille de quiz — second modèle supervisé (classification multiclasses).

Problème : ~40 % des vidéos concurrentes n'ont pas de famille (les règles ne matchent pas).
On apprend à prédire la famille à partir du TITRE, puis on comble les valeurs manquantes.

Démarche (comme le scoring) : baseline -> régression logistique -> LightGBM,
validation croisée, métriques adaptées au déséquilibre (macro-F1), matrice de confusion.

Usage :
  python -m src.family.run_family
  python -m src.family.run_family --min-count 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import joblib  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)
from sklearn.model_selection import cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .. import taxonomy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("family")

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "data" / "raw" / "competitors" / "videos.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"


def _text(df: pd.DataFrame) -> pd.Series:
    return (df["title"].fillna("") + " " + df["description"].fillna("")).str.strip()


def _pipeline(clf) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
        ("clf", clf),
    ])


def run(min_count: int = 5) -> None:
    df = pd.read_csv(VIDEOS)
    labeled = df[df["family"].notna()].copy()
    labeled["text"] = _text(labeled)

    # on garde les familles assez représentées pour un apprentissage fiable
    counts = labeled["family"].value_counts()
    keep = counts[counts >= min_count].index
    dropped = counts[counts < min_count]
    data = labeled[labeled["family"].isin(keep)]
    log.info("Familles gardées (>=%d) : %s", min_count, dict(counts[keep]))
    if len(dropped):
        log.info("Familles trop rares écartées : %s", dict(dropped))

    X, y = data["text"], data["family"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=42)

    candidates = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "lgbm": LGBMClassifier(class_weight="balanced", n_estimators=200,
                               learning_rate=0.05, num_leaves=15, verbose=-1),
    }

    results, fitted = {}, {}
    for name, clf in candidates.items():
        pipe = _pipeline(clf)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results[name] = {
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y_te, pred, average="weighted"), 4),
        }
        fitted[name] = pipe
        log.info("%-7s → %s", name, results[name])

    # meilleur = macro-F1 (robuste au déséquilibre)
    best_name = max((n for n in results if n != "dummy"),
                    key=lambda n: results[n]["macro_f1"])
    best = fitted[best_name]
    log.info("Meilleur modèle : %s", best_name)

    # validation croisée (macro-F1) du meilleur
    cv = cross_val_score(_pipeline(candidates[best_name]), X, y, cv=5, scoring="f1_macro")
    log.info("CV macro-F1 (%s) : %.3f ± %.3f", best_name, cv.mean(), cv.std())

    # rapport détaillé + matrice de confusion
    pred = best.predict(X_te)
    report = classification_report(y_te, pred, zero_division=0, output_dict=True)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_te, pred, labels=labels)
    _plot_confusion(cm, labels)

    # sauvegardes
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODELS / "family_classifier.joblib")
    (REPORTS / "family_metrics.json").write_text(json.dumps({
        "best": best_name, "results": results,
        "cv_macro_f1_mean": round(float(cv.mean()), 4),
        "cv_macro_f1_std": round(float(cv.std()), 4),
        "classes": labels, "classification_report": report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("→ models/family_classifier.joblib · reports/family_metrics.json")

    # combler les familles manquantes
    _fill_unknown(df, best)


def _plot_confusion(cm, labels) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set(title="Classifieur de famille — matrice de confusion",
           xlabel="prédit", ylabel="réel")
    fig.tight_layout()
    fig.savefig(FIG / "08_family_confusion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fill_unknown(df: pd.DataFrame, model) -> None:
    unknown = df[df["family"].isna()].copy()
    if unknown.empty:
        log.info("Aucune famille manquante à combler.")
        return
    unknown["text"] = _text(unknown)
    unknown["family_predicted"] = model.predict(unknown["text"])
    out = unknown[["video_id", "title", "family_predicted"]]
    out.to_csv(ROOT / "data" / "processed" / "family_predictions.csv", index=False)
    log.info("→ data/processed/family_predictions.csv : %d familles comblées", len(out))
    log.info("Répartition prédite : %s", dict(out["family_predicted"].value_counts()))


def main() -> None:
    p = argparse.ArgumentParser(description="Classifieur de famille de quiz")
    p.add_argument("--min-count", type=int, default=5,
                   help="taille minimale d'une famille pour être apprise")
    args = p.parse_args()
    run(min_count=args.min_count)


if __name__ == "__main__":
    main()
