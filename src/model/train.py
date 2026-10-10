"""Phase 9: train, tune and compare models on the 2017 validation set, then calibrate and save the best.

Run from the repo root, after `python -m src.dataset.features`:
    python -m src.model.train

The 2018 test set is NOT touched here. It is used once, later, by `python -m src.model.final_test`.

Steps:
1. For each data version (main, with_covid) x feature set (strict, strict+caution) x model,
   try every setting in PARAM_GRIDS: fit on 2010-2016, score on 2017 (main validation rows,
   and also the with_covid validation rows).
2. Pick the run with the best validation PR-AUC (within the model size budget).
3. Choose a calibrator by cross-validated Brier score within 2017, pick the F1 threshold,
   set the Low/Medium/High bands from the 2017 base rate, and measure feature importance.
4. Save the model (unless one risk band holds almost every trial) and write the reports.
"""

import hashlib
import io
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from src.config import (
    BAND_MAX_SHARE_PCT,
    CALIBRATION_BINS,
    CALIBRATION_FOLDS,
    DATA_VERSIONS,
    FEATURE_SETS,
    MODEL_MAX_SIZE_MB,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PARAM_GRIDS,
    PERMUTATION_REPEATS,
    RANDOM_SEED,
    REPORTS_DIR,
    RISK_BAND_MULTIPLIERS,
    SPLIT_YEARS,
    VALIDATION_REPORT_JSON,
    VALIDATION_REPORT_MD,
)
from src.dataset.features import FEATURES, feature_names, load_features
from src.model.calibration import CalibratedModel, Calibrator, compare_methods
from src.model.metrics import (
    all_scores,
    band_cutoffs,
    band_table,
    best_f1_threshold,
    labels_of,
    ranking_scores,
    reliability_table,
    threshold_scores,
)
from src.model.pipeline import MODEL_NAMES, make_pipeline

DISCLAIMER = ("Research and portfolio tool. The risk score describes patterns in past registry data; "
              "it is not medical, legal or investment advice and does not judge any specific trial.")


# --- small helpers ------------------------------------------------------------------

def size_mb(obj: object) -> float:
    """Size of an object when saved with joblib (compressed, the same way the model is saved)."""
    buffer = io.BytesIO()
    joblib.dump(obj, buffer, compress=3)
    return round(len(buffer.getvalue()) / 1e6, 2)


def file_sha256(path) -> str:
    """Fingerprint of a file, so reports can prove which model file they describe."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band_check(bands: list[dict], max_share_pct: float) -> dict:
    """Is any risk band holding more than max_share_pct of the trials?"""
    largest = max(bands, key=lambda b: b["share_pct"])
    return {"largest_band": largest["band"], "largest_share_pct": largest["share_pct"],
            "max_allowed_pct": max_share_pct, "ok": largest["share_pct"] <= max_share_pct}


# --- step 1: the grid -------------------------------------------------------------------

def run_one(model_name: str, params: dict, features: list[str], train: pd.DataFrame,
            val_main: pd.DataFrame, val_covid: pd.DataFrame) -> tuple[dict, Pipeline]:
    """Fit one setting on the training years and score it on both validation row sets."""
    pipeline = make_pipeline(model_name, params, features)
    start = time.perf_counter()
    pipeline.fit(train[features], labels_of(train))
    fit_seconds = time.perf_counter() - start

    y_main = labels_of(val_main)
    p_main = pipeline.predict_proba(val_main[features])[:, 1]
    p_covid = pipeline.predict_proba(val_covid[features])[:, 1]
    at_best_f1 = (threshold_scores(y_main, p_main, best_f1_threshold(y_main, p_main))
                  if model_name != "dummy" else threshold_scores(y_main, p_main, 0.5))
    run = {
        "model": model_name,
        "params": params,
        "val_main": ranking_scores(y_main, p_main),
        "val_main_at_best_f1": at_best_f1,
        "val_with_covid": ranking_scores(labels_of(val_covid), p_covid),
        "fit_seconds": round(fit_seconds, 1),
        "size_mb": size_mb(pipeline),
    }
    return run, pipeline


def run_grid(val_main: pd.DataFrame, val_covid: pd.DataFrame) -> tuple[list[dict], dict, Pipeline]:
    """Every data version x feature set x model x setting. Returns (runs, best run, best pipeline).

    The best run is the highest main-validation PR-AUC among real models within the size budget.
    """
    runs: list[dict] = []
    best_run, best_pipeline = None, None
    for version in DATA_VERSIONS:
        train = load_features(("train",), version)
        for set_name, groups in FEATURE_SETS.items():
            features = feature_names(groups)
            for model_name in MODEL_NAMES:
                for params in ParameterGrid(PARAM_GRIDS[model_name]):
                    run, pipeline = run_one(model_name, params, features, train, val_main, val_covid)
                    run.update({"data_version": version, "feature_set": set_name, "train_rows": len(train)})
                    runs.append(run)
                    print(f"{version:10s} {set_name:15s} {model_name:13s} {str(params):70s} "
                          f"PR-AUC {run['val_main']['pr_auc']:.4f}  ROC-AUC {run['val_main']['roc_auc']:.4f}  "
                          f"{run['fit_seconds']:6.1f}s  {run['size_mb']:6.2f} MB", flush=True)
                    eligible = model_name != "dummy" and run["size_mb"] <= MODEL_MAX_SIZE_MB
                    if eligible and (best_run is None or run["val_main"]["pr_auc"] > best_run["val_main"]["pr_auc"]):
                        best_run, best_pipeline = run, pipeline
    return runs, best_run, best_pipeline


def best_per_group(runs: list[dict]) -> list[dict]:
    """The best setting (by main-validation PR-AUC) for each data version x feature set x model."""
    best: dict[tuple, dict] = {}
    for run in runs:
        key = (run["data_version"], run["feature_set"], run["model"])
        if key not in best or run["val_main"]["pr_auc"] > best[key]["val_main"]["pr_auc"]:
            best[key] = run
    return list(best.values())


# --- step 3: calibration, threshold, bands, importance ---------------------------------

def feature_importance(pipeline: Pipeline, features: list[str], val: pd.DataFrame) -> list[dict]:
    """Permutation importance: how much validation PR-AUC drops when one feature is shuffled."""
    result = permutation_importance(pipeline, val[features], labels_of(val), scoring="average_precision",
                                    n_repeats=PERMUTATION_REPEATS, random_state=RANDOM_SEED, n_jobs=1)
    rows = [{"feature": name, "pr_auc_drop_mean": round(float(m), 5), "pr_auc_drop_std": round(float(s), 5)}
            for name, m, s in zip(features, result.importances_mean, result.importances_std)]
    return sorted(rows, key=lambda r: r["pr_auc_drop_mean"], reverse=True)


def logreg_coefficients(runs: list[dict], version: str, set_name: str, top: int = 15) -> dict:
    """Refit the best Logistic Regression for this data version and feature set; return its largest coefficients.

    Inputs are scaled, so coefficients are comparable: positive = more likely terminated.
    """
    best = max((r for r in runs if r["model"] == "logreg" and r["data_version"] == version
                and r["feature_set"] == set_name), key=lambda r: r["val_main"]["pr_auc"])
    features = feature_names(FEATURE_SETS[set_name])
    train = load_features(("train",), version)
    pipeline = make_pipeline("logreg", best["params"], features).fit(train[features], labels_of(train))
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    order = np.argsort(-np.abs(coefs))[:top]
    return {"params": best["params"],
            "top": [{"input": str(names[i]), "coefficient": round(float(coefs[i]), 4)} for i in order]}


def calibrate_and_score(pipeline: Pipeline, features: list[str], val_main: pd.DataFrame,
                        val_covid: pd.DataFrame) -> dict:
    """Choose a calibrator, the F1 threshold and the risk bands, all from 2017 validation data.

    Validation numbers after calibration use out-of-fold probabilities (see calibration.py).
    """
    y = labels_of(val_main)
    raw = pipeline.predict_proba(val_main[features])[:, 1]
    methods = compare_methods(raw, y, CALIBRATION_FOLDS)
    method = min(methods, key=lambda m: methods[m]["brier_cv"])
    calibrated = methods[method]["out_of_fold"]
    calibrator = Calibrator(method).fit(raw, y)

    threshold = best_f1_threshold(y, calibrated)
    cutoffs = band_cutoffs(float(y.mean()), RISK_BAND_MULTIPLIERS)
    bands = band_table(y, calibrated, cutoffs)
    covid_p = calibrator.transform(pipeline.predict_proba(val_covid[features])[:, 1])
    return {
        "calibrator": calibrator,
        "calibration": {
            "method": method,
            "brier_cv_by_method": {m: v["brier_cv"] for m, v in methods.items()},
            "reliability_raw": reliability_table(y, raw, CALIBRATION_BINS),
            "reliability_calibrated": reliability_table(y, calibrated, CALIBRATION_BINS),
        },
        "threshold": threshold,
        "val_main": all_scores(y, calibrated, threshold),
        "val_with_covid": all_scores(labels_of(val_covid), covid_p, threshold),  # calibrator fitted on main rows
        "dummy_val_main": all_scores(y, np.zeros(len(y)), 0.5),
        "band_cutoffs": cutoffs,
        "bands_val": bands,
        "band_check": band_check(bands, BAND_MAX_SHARE_PCT),
    }


# --- step 4: save ---------------------------------------------------------------------

def build_metadata(best: dict, features: list[str], result: dict, model_size: float, sha256: str) -> dict:
    """Everything needed to use and describe the saved model."""
    import lightgbm
    import sklearn

    groups = {f.name: f.group for f in FEATURES}
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": best["model"],
        "params": best["params"],
        "data_version": best["data_version"],
        "feature_set": best["feature_set"],
        "features": [{"name": n, "group": groups[n]} for n in features],
        "trained_on_start_years": list(SPLIT_YEARS["train"]),
        "tuned_on_start_years": list(SPLIT_YEARS["val"]),
        "train_rows": best["train_rows"],
        "calibration_method": result["calibration"]["method"],
        "f1_threshold": round(result["threshold"], 4),
        "risk_bands": {"cutoffs": result["band_cutoffs"], "multipliers_of_base_rate": list(RISK_BAND_MULTIPLIERS),
                       "validation_base_rate": result["val_main"]["base_rate"]},
        "validation_scores": result["val_main"],
        "file": MODEL_PATH.name,
        "size_mb": model_size,
        "sha256": sha256,
        "library_versions": {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__},
        "disclaimer": DISCLAIMER,
    }


def main() -> None:
    """Run the whole training step and write the reports."""
    from src.model.card import write_model_card
    from src.model.report import write_validation_report

    started = time.perf_counter()
    val_main = load_features(("val",), "main")
    val_covid = load_features(("val",), "with_covid")
    print(f"Validation rows: main {len(val_main)}, with_covid {len(val_covid)}")

    runs, best, pipeline = run_grid(val_main, val_covid)
    features = feature_names(FEATURE_SETS[best["feature_set"]])
    print(f"\nChosen: {best['model']} {best['params']} ({best['data_version']}, {best['feature_set']})")

    result = calibrate_and_score(pipeline, features, val_main, val_covid)
    print("Measuring feature importance...")
    importance = feature_importance(pipeline, features, val_main)
    coefficients = logreg_coefficients(runs, best["data_version"], best["feature_set"])

    saved = result["band_check"]["ok"]
    model_size, sha256 = None, None
    if saved:
        model = CalibratedModel(pipeline, result["calibrator"], features)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH, compress=3)
        model_size = round(MODEL_PATH.stat().st_size / 1e6, 2)
        sha256 = file_sha256(MODEL_PATH)
        metadata = build_metadata(best, features, result, model_size, sha256)
        MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {MODEL_PATH} ({model_size} MB) and {MODEL_METADATA_PATH.name}")
    else:
        check = result["band_check"]
        print(f"NOT SAVED: the {check['largest_band']} band holds {check['largest_share_pct']}% of validation "
              f"trials (limit {check['max_allowed_pct']}%). Discuss the bands before saving.")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_minutes": round((time.perf_counter() - started) / 60, 1),
        "validation_rows": {"main": len(val_main), "with_covid": len(val_covid)},
        "selection_rule": "highest PR-AUC on the main 2017 validation rows, among non-dummy runs "
                          f"no larger than {MODEL_MAX_SIZE_MB} MB",
        "chosen": {k: best[k] for k in ("model", "params", "data_version", "feature_set", "train_rows",
                                        "fit_seconds", "size_mb")},
        "saved": saved, "model_size_mb": model_size, "model_sha256": sha256,
        **{k: v for k, v in result.items() if k != "calibrator"},
        "feature_importance": importance,
        "logreg_coefficients": coefficients,
        "best_per_group": best_per_group(runs),
        "runs": runs,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written: {write_validation_report(report)}")
    if saved:
        print(f"Model card written: {write_model_card()}")
    print(f"Total runtime: {report['runtime_minutes']} minutes")


if __name__ == "__main__":
    main()
