"""Phase 9: the ONE evaluation of the saved model on the 2018 test set (plus the 2019-2020 recent set).

Run from the repo root, only after the chosen model has been approved:
    python -m src.model.final_test --approved

Guards:
- refuses to run without --approved,
- refuses to run a second time (if the final test report already exists),
- refuses if the model file is not the one described in model_metadata.json (sha256 check).
The F1 threshold and risk bands come from the metadata (fixed on 2017 validation data).
"""

import argparse
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_VERSIONS, FINAL_TEST_REPORT_JSON, MODEL_METADATA_PATH, MODEL_PATH
from src.dataset.features import load_features
from src.model.calibration import CalibratedModel
from src.model.metrics import all_scores, band_table, labels_of
from src.model.report import censoring_note, write_final_test_report
from src.model.train import file_sha256


def ensure_first_run(report_path: Path) -> None:
    """Raise if the final test was already run (its report exists)."""
    if report_path.exists():
        raise RuntimeError(f"The final test was already run ({report_path} exists). The test set is used once; "
                           "running it again after changes would turn it into a tuning set.")


def verify_model_file(model_path: Path, metadata: dict) -> str:
    """Raise if the model file differs from the one the metadata describes. Returns its sha256."""
    sha256 = file_sha256(model_path)
    if sha256 != metadata["sha256"]:
        raise RuntimeError("models/trial_risk_model.joblib does not match model_metadata.json (sha256 differs). "
                           "Re-run python -m src.model.train.")
    return sha256


def evaluate_rows(model: CalibratedModel, metadata: dict, rows: pd.DataFrame) -> dict:
    """Scores of the model and of the dummy on one set of rows, plus the risk-band table."""
    y = labels_of(rows)
    p = model.predict_proba(rows)
    cutoffs = metadata["risk_bands"]["cutoffs"]
    return {
        "rows": len(rows),
        "model": all_scores(y, p, metadata["f1_threshold"]),
        "dummy": all_scores(y, np.zeros(len(y)), 0.5),
        "bands": band_table(y, p, cutoffs),
    }


def run_final_test(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    report_path: Path = FINAL_TEST_REPORT_JSON,
    load: Callable[..., pd.DataFrame] = load_features,
    note: str | None = None,
) -> dict:
    """Evaluate the saved model once on test and recent rows, for both data versions. Returns the report.

    `note` is the censoring note for the recent set (read from the Phase 8 audit if not given).
    """
    ensure_first_run(report_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sha256 = verify_model_file(model_path, metadata)
    model: CalibratedModel = joblib.load(model_path)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": metadata["model"], "data_version": metadata["data_version"],
        "feature_set": metadata["feature_set"], "model_sha256": sha256,
        "threshold": metadata["f1_threshold"], "band_cutoffs": metadata["risk_bands"]["cutoffs"],
        "censoring_note": note if note is not None else censoring_note(),
        "test": {}, "recent": {},
    }
    for part in ("test", "recent"):
        for version in DATA_VERSIONS:
            rows = load((part,), version, final_evaluation=True)
            report[part][version] = evaluate_rows(model, metadata, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Command-line entry point (needs --approved)."""
    from src.model.card import write_model_card

    parser = argparse.ArgumentParser(description="One-time final evaluation on the 2018 test set.")
    parser.add_argument("--approved", action="store_true",
                        help="confirm that the chosen model was approved for the one final test run")
    args = parser.parse_args()
    if not args.approved:
        print("Not run: add --approved once the chosen model has been approved. The test set is used only once.")
        raise SystemExit(1)
    try:
        report = run_final_test()
    except RuntimeError as err:
        print(err)
        raise SystemExit(1)
    print(f"Final test report written: {write_final_test_report(report)}")
    print(f"Model card updated: {write_model_card()}")


if __name__ == "__main__":
    main()
