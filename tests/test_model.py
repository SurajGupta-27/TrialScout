"""Tests for Phase 9 (training, calibration, bands, saving, final-test guard). Offline, synthetic data.

Run from the repo root:  python -m tests.test_model
"""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.dataset.features import FEATURES, feature_names
from src.model.calibration import METHODS, CalibratedModel, Calibrator, compare_methods
from src.model.final_test import ensure_first_run, run_final_test, verify_model_file
from src.model.metrics import (
    assign_bands,
    band_cutoffs,
    band_table,
    best_f1_threshold,
    ranking_scores,
    reliability_table,
    threshold_scores,
)
from src.model.pipeline import MODEL_NAMES, columns_by_kind, make_pipeline
from src.model.train import band_check, build_metadata, file_sha256, size_mb

# Tiny, fast settings for each model (the real grids live in config.py).
TINY_PARAMS = {"dummy": {}, "logreg": {"C": 1.0}, "random_forest": {"max_depth": 3},
               "lightgbm": {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 5}}


def fake_table(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A synthetic feature table with every feature; termination is likelier for PHASE1 and many sites."""
    rng = np.random.default_rng(seed)
    data = {}
    for f in FEATURES:
        if f.kind == "category":
            data[f.name] = rng.choice(["A", "B", "MISSING"], n)
        elif f.kind == "number":
            values = rng.integers(0, 50, n).astype(float)
            values[rng.random(n) < 0.1] = np.nan
            data[f.name] = values
        else:
            data[f.name] = rng.random(n) < 0.3
    table = pd.DataFrame(data)
    table["phase"] = rng.choice(["PHASE1", "PHASE3"], n)
    risk = 0.05 + 0.4 * (table["phase"] == "PHASE1") + 0.004 * table["n_locations"].fillna(0)
    table["label"] = (rng.random(n) < risk).astype(int)
    table["nct_id"] = [f"NCT{i:08d}" for i in range(n)]
    return table


# --- pipeline -------------------------------------------------------------------------

def test_every_feature_has_a_preprocessing_route() -> None:
    """Each feature goes to exactly one transformer."""
    names = feature_names(("strict", "caution"))
    cols = columns_by_kind(names)
    routed = [n for group in cols.values() for n in group]
    assert sorted(routed) == sorted(names)


def test_pipelines_fit_and_predict() -> None:
    """All four models train and give probabilities in [0, 1]; unseen categories don't crash."""
    train, new = fake_table(seed=1), fake_table(n=50, seed=2)
    new.loc[0, "masking"] = "NEVER_SEEN_VALUE"
    features = feature_names(("strict", "caution"))
    for name in MODEL_NAMES:
        pipeline = make_pipeline(name, TINY_PARAMS[name], features).fit(train[features], train["label"])
        p = pipeline.predict_proba(new[features])[:, 1]
        assert p.shape == (50,) and np.all((p >= 0) & (p <= 1)), name


def test_learned_signal_beats_dummy() -> None:
    """On data with a planted pattern, Logistic Regression ranks better than the dummy."""
    train, val = fake_table(n=2000, seed=3), fake_table(n=1000, seed=4)
    features = feature_names(("strict", "caution"))
    y = val["label"].to_numpy()
    scores = {}
    for name in ("dummy", "logreg"):
        pipeline = make_pipeline(name, TINY_PARAMS[name], features).fit(train[features], train["label"])
        scores[name] = ranking_scores(y, pipeline.predict_proba(val[features])[:, 1])
    assert scores["dummy"]["roc_auc"] == 0.5
    assert abs(scores["dummy"]["pr_auc"] - y.mean()) < 1e-3  # no skill = base rate
    assert scores["logreg"]["roc_auc"] > 0.7 and scores["logreg"]["pr_auc"] > scores["dummy"]["pr_auc"]


# --- metrics, calibration, bands ------------------------------------------------------

def test_threshold_and_confusion_matrix() -> None:
    """Best-F1 threshold separates a clean example; the confusion matrix adds up."""
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = best_f1_threshold(y, p)
    assert 0.3 < threshold <= 0.8
    s = threshold_scores(y, p, threshold)
    assert s["f1"] == 1.0 and s["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 2}
    nothing = threshold_scores(y, np.zeros(5), 0.5)  # the dummy: never predicts terminated
    assert nothing["recall"] == 0.0 and nothing["precision"] == 0.0 and nothing["accuracy"] == 0.6


def test_reliability_table() -> None:
    """Bins have equal sizes and cover every trial."""
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    rows = reliability_table((rng.random(1000) < p).astype(int), p, 10)
    assert len(rows) == 10 and sum(r["trials"] for r in rows) == 1000
    assert rows[0]["mean_predicted"] < rows[-1]["mean_predicted"]


def test_calibration_methods() -> None:
    """Every method gives probabilities in [0, 1]; calibration fixes inflated scores."""
    rng = np.random.default_rng(0)
    true_p = rng.random(3000) * 0.3
    y = (rng.random(3000) < true_p).astype(int)
    inflated = np.clip(true_p * 2.5, 0, 1)  # e.g. what class weighting does
    results = compare_methods(inflated, y, folds=5)
    assert set(results) == set(METHODS)
    for method in METHODS:
        out = results[method]["out_of_fold"]
        assert out.shape == (3000,) and np.all((out >= 0) & (out <= 1))
    assert results["sigmoid"]["brier_cv"] < results["none"]["brier_cv"]
    assert np.allclose(Calibrator("none").fit(inflated, y).transform(inflated), inflated)


def test_risk_bands() -> None:
    """Low < b <= Medium < 2b <= High, and the band table adds up."""
    cutoffs = band_cutoffs(0.12, (1.0, 2.0))
    assert cutoffs == {"low_max": 0.12, "high_min": 0.24}
    p = np.array([0.05, 0.119, 0.12, 0.2, 0.24, 0.9])
    assert list(assign_bands(p, cutoffs)) == ["Low", "Low", "Medium", "Medium", "High", "High"]
    table = band_table(np.array([0, 0, 0, 1, 1, 1]), p, cutoffs)
    assert [b["trials"] for b in table] == [2, 2, 2]
    assert abs(sum(b["share_pct"] for b in table) - 100) < 0.2
    assert table[2]["actual_rate_pct"] == 100.0


def test_band_check_flags_one_dominant_band() -> None:
    """If one band holds more than the limit, saving is blocked."""
    ok = [{"band": "Low", "share_pct": 55.0}, {"band": "Medium", "share_pct": 30.0}, {"band": "High", "share_pct": 15.0}]
    bad = [{"band": "Low", "share_pct": 95.0}, {"band": "Medium", "share_pct": 4.0}, {"band": "High", "share_pct": 1.0}]
    assert band_check(ok, 80.0)["ok"]
    assert not band_check(bad, 80.0)["ok"] and band_check(bad, 80.0)["largest_band"] == "Low"


# --- saving and the final-test guards ---------------------------------------------------

def save_fake_model(folder: Path) -> tuple[Path, Path, dict]:
    """Train a tiny calibrated model, save it with metadata like train.py does."""
    features = feature_names(("strict",))
    train = fake_table(seed=5)
    pipeline = make_pipeline("logreg", {"C": 1.0}, features).fit(train[features], train["label"])
    raw = pipeline.predict_proba(train[features])[:, 1]
    model = CalibratedModel(pipeline, Calibrator("sigmoid").fit(raw, train["label"].to_numpy()), features)
    model_path, metadata_path = folder / "model.joblib", folder / "meta.json"
    joblib.dump(model, model_path, compress=3)
    best = {"model": "logreg", "params": {"C": 1.0}, "data_version": "main", "feature_set": "strict",
            "train_rows": len(train)}
    result = {"calibration": {"method": "sigmoid"}, "threshold": 0.3,
              "band_cutoffs": {"low_max": 0.12, "high_min": 0.24}, "val_main": {"base_rate": 0.12}}
    metadata = build_metadata(best, features, result, size_mb(model), file_sha256(model_path))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return model_path, metadata_path, metadata


def test_metadata_is_complete_and_model_is_small() -> None:
    """Saved metadata has what Phase 10 and the model card need; the model loads and predicts."""
    with tempfile.TemporaryDirectory() as tmp:
        model_path, _, meta = save_fake_model(Path(tmp))
        for key in ("model", "params", "features", "trained_on_start_years", "calibration_method",
                    "f1_threshold", "risk_bands", "sha256", "size_mb", "library_versions", "disclaimer"):
            assert key in meta, key
        assert meta["trained_on_start_years"] == [2010, 2016] and meta["size_mb"] < 1
        model = joblib.load(model_path)
        p = model.predict_proba(fake_table(n=20, seed=6))
        assert p.shape == (20,) and np.all((p >= 0) & (p <= 1))


def test_final_test_runs_once_and_checks_the_model_file() -> None:
    """The final test refuses a second run and a model file that doesn't match its metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        model_path, metadata_path, meta = save_fake_model(folder)
        report_path = folder / "final.json"
        calls = []

        def fake_load(parts, version, final_evaluation=False):
            calls.append((parts, version, final_evaluation))
            return fake_table(n=100, seed=7)

        report = run_final_test(model_path, metadata_path, report_path, load=fake_load, note="note")
        assert report_path.exists() and set(report["test"]) == {"main", "with_covid"}
        assert all(final for _, _, final in calls) and len(calls) == 4  # test + recent, two versions
        assert report["test"]["main"]["dummy"]["roc_auc"] == 0.5
        try:
            ensure_first_run(report_path)
            raise AssertionError("expected RuntimeError on a second run")
        except RuntimeError as err:
            assert "already run" in str(err)

        model_path.write_bytes(model_path.read_bytes() + b"tampered")
        try:
            verify_model_file(model_path, meta)
            raise AssertionError("expected RuntimeError for a changed model file")
        except RuntimeError as err:
            assert "sha256" in str(err)


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} model tests passed.")
