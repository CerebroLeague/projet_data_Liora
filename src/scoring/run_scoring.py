"""
Orchestrateur du Sprint C — entraînement & évaluation du scoring viral.

Pipeline :
  1. Charge data/processed/dataset.csv, sépare X/y (anti-fuite), split train/test
  2. Évalue baselines (Dummy, Linéaire) + LightGBM par défaut
  3. Optimise LightGBM (GridSearchCV, scorer = Spearman)
  4. Réévalue le meilleur modèle sur le test
  5. Interprétabilité SHAP
  6. Sauvegarde le modèle (models/) + les métriques (reports/)

Usage :
  python -m src.scoring.run_scoring
  python -m src.scoring.run_scoring --no-grid --no-shap   # rapide
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib

from . import dataset, models
from .metrics import evaluate

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("scoring")

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def run(*, do_grid: bool = True, do_shap: bool = True) -> None:
    X, y = dataset.load_xy()
    X_tr, X_te, y_tr, y_te = dataset.split(X, y)
    log.info("Dataset : %d lignes, %d features | train=%d test=%d",
             len(X), X.shape[1], len(X_tr), len(X_te))

    results: dict[str, dict] = {}

    # 1-2. baselines + LightGBM par défaut
    contenders = {**models.make_baselines(), "lgbm_default": models.make_lgbm()}
    for name, model in contenders.items():
        model.fit(X_tr, y_tr)
        results[name] = evaluate(y_te, model.predict(X_te))
        log.info("%-14s → %s", name, results[name])

    fitted = {"lgbm_default": contenders["lgbm_default"]}

    # 3-4. optimisation
    if do_grid:
        gs = models.grid_search(X_tr, y_tr)
        results["lgbm_tuned"] = evaluate(y_te, gs.predict(X_te))
        log.info("lgbm_tuned     → %s | params=%s", results["lgbm_tuned"], gs.best_params_)
        fitted["lgbm_tuned"] = gs.best_estimator_

    # on déploie le meilleur LightGBM sur le test (petit volume => on garde le + performant)
    best_name = max(fitted, key=lambda n: results[n]["spearman"])
    best_model = fitted[best_name]

    # 5. SHAP
    if do_shap:
        from .interpret import explain
        importance = explain(best_model, X_te)
        log.info("Top features (SHAP) :")
        for _, r in importance.head(8).iterrows():
            log.info("   %-22s %.4f", r["feature"], r["mean_abs_shap"])
        importance.to_csv(REPORTS / "shap_importance.csv", index=False)

    # 6. sauvegarde
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS / "scoring_viral.joblib")
    # colonnes de features attendues par le modèle (pour scorer les idées générées, Sprint D)
    (MODELS / "feature_columns.json").write_text(
        json.dumps(list(X.columns), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS / "scoring_metrics.json").write_text(
        json.dumps({"best": best_name, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("→ models/scoring_viral.joblib (%s)", best_name)
    log.info("→ reports/scoring_metrics.json")

    # verdict H2
    rho = results[best_name]["spearman"]
    verdict = "VALIDÉE ✔" if rho > 0.5 else "non atteinte (à améliorer)"
    log.info("Hypothèse H2 (Spearman > 0,5) : %.3f → %s", rho, verdict)


def main() -> None:
    p = argparse.ArgumentParser(description="Scoring viral (Sprint C)")
    p.add_argument("--no-grid", action="store_true", help="sauter l'optimisation GridSearch")
    p.add_argument("--no-shap", action="store_true", help="sauter l'interprétabilité SHAP")
    args = p.parse_args()
    run(do_grid=not args.no_grid, do_shap=not args.no_shap)


if __name__ == "__main__":
    main()
