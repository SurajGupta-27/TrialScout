"""Phase 8 leakage audit: which columns may be features, which rows were removed, and the split.

Called by `python -m src.dataset.features`. Every number comes from the data or from
live API counts saved to data/censoring_counts.json; nothing is typed by hand.

Outputs (small, committed to git):
    data/LEAKAGE_AUDIT.json   all numbers, machine-readable
    data/LEAKAGE_AUDIT.md     the same as readable tables
    data/censoring_counts.json  API counts of still-running trials per start year

Which data each check may look at:
- Label-based checks (single-feature AUC, missing values by label) use the TRAINING years only.
- Drift compares training with validation (2017).
- The 2018 test set is only counted for the class-balance table. Its labels are never analysed.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    AUDIT_AUC_REVIEW_THRESHOLD,
    AUDIT_DRIFT_REVIEW_THRESHOLD,
    CENSORING_COUNTS_PATH,
    DATA_DIR,
    DATASET_REQUEST_DELAY_SECONDS,
    SPLIT_YEARS,
)
from src.dataset.columns import COLUMNS
from src.dataset.features import (
    EXCLUDED_COLUMNS,
    FEATURES,
    MANUAL_REVIEW_NOTES,
    MISSING,
    VERSIONS,
    Feature,
    feature_names,
    select_rows,
)
from src.dataset.filters import (
    LATE_REGISTRATION_REASON,
    covid_terminated,
    registered_after_primary_completion,
    trial_end_date,
)
from src.dataset.report import pct

AUDIT_JSON = DATA_DIR / "LEAKAGE_AUDIT.json"
AUDIT_MD = DATA_DIR / "LEAKAGE_AUDIT.md"

# Result of reading the flagged trials whose primary completion date is before 2020 (Phase 8).
COVID_PRE_2020_CHECK = (
    "Manual check of the flagged trials with a primary completion date before 2020: their why_stopped "
    "texts really do give COVID-19 or the pandemic as a reason (e.g. \"suspended in March 2020 due to "
    "COVID\"). For a terminated trial the registry often records the last participant's date as the end "
    "date, before the pause, so these are true COVID stops, not keyword false positives."
)

# Statuses meaning "still running" (no final outcome yet). UNKNOWN is counted separately:
# the record has not been verified for 2+ years, so we can't tell if it is running.
ONGOING_STATUSES = "RECRUITING|NOT_YET_RECRUITING|ACTIVE_NOT_RECRUITING|ENROLLING_BY_INVITATION|SUSPENDED"

# Leakage columns measured anyway, to show what they would have done (never used as features).
# Each maps to a function that computes the column from trials.parquet rows.
MEASURED_LEAKS = {
    "enrollment_count": lambda t: pd.to_numeric(t["enrollment_count"], errors="coerce").astype("float64"),
    "registration delay (first_submit_date - start_date, days)": lambda t: (
        (pd.to_datetime(t["first_submit_date"]) - pd.to_datetime(t["start_date"])).dt.days.astype("float64")
    ),
    "has_results": lambda t: t["has_results"].astype("boolean").fillna(False).astype(float),
}


# --- right-censoring: live API counts ---------------------------------------------

def fetch_censoring_counts(years: range) -> dict:
    """Ask the API, per start year, how many interventional trials exist and how many are still running."""
    # Imported here so the offline tests never need the network modules.
    from src.dataset.download import count_studies, make_fetcher, make_session, make_snapshot_getter

    session = make_session()
    fetch = make_fetcher(session)
    by_year = {}
    for year in years:
        counts = {}
        for name, status in (("all", ""), ("ongoing", ONGOING_STATUSES), ("unknown", "UNKNOWN")):
            counts[name] = count_studies(fetch, status, year, year)
            time.sleep(DATASET_REQUEST_DELAY_SECONDS)
        by_year[str(year)] = counts
        print(f"{year}: {counts}")
    return {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "registry_snapshot": make_snapshot_getter(session)(),
        "ongoing_statuses": ONGOING_STATUSES.split("|"),
        "by_year": by_year,
    }


def load_censoring_counts(years: range, path: Path = CENSORING_COUNTS_PATH) -> dict:
    """Read saved API counts, or fetch and save them the first time (needs internet, ~1 minute)."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print("Fetching still-running trial counts per start year from the API...")
    counts = fetch_censoring_counts(years)
    path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


def censoring_table(trials: pd.DataFrame, counts: dict) -> list[dict]:
    """Per start year: how complete the year is, and whether its finished trials look different.

    Uses all finished trials (before the late-registration filter), so the counts
    compare like with like against the API totals.
    """
    duration_years = (trial_end_date(trials) - pd.to_datetime(trials["start_date"])).dt.days / 365.25
    rows = []
    for year, group in trials.groupby("start_year"):
        api = counts["by_year"].get(str(int(year)))
        if not api:
            continue
        durations = duration_years[group.index]
        within_5 = group[durations <= 5]
        rows.append({
            "start_year": int(year),
            "all_trials": api["all"],
            "still_running_pct": pct(api["ongoing"], api["all"]),
            "unknown_status_pct": pct(api["unknown"], api["all"]),
            "finished_pct": pct(len(group), api["all"]),
            "finished_lasting_over_5y_pct": pct(int((durations > 5).sum()), len(group)),
            "terminated_pct_if_finished_within_5y": pct(int(within_5["label"].sum()), len(within_5)),
        })
    return rows


# --- rows removed or flagged, and the split -----------------------------------------

def cleaning_summary(trials: pd.DataFrame) -> dict:
    """Counts for the late-registration filter and the COVID flag."""
    late = registered_after_primary_completion(trials)
    kept = trials[~late]
    covid = covid_terminated(kept)
    no_end_date = trial_end_date(trials).isna()
    ended_before_covid = trial_end_date(kept) < pd.Timestamp("2020-01-01")
    return {
        "trials_in": len(trials),
        "late_registration": {
            "reason": LATE_REGISTRATION_REASON,
            "dropped": int(late.sum()),
            "terminated_among_dropped": int(trials.loc[late, "label"].sum()),
            "terminated_pct_among_dropped": pct(int(trials.loc[late, "label"].sum()), int(late.sum())),
            "kept_without_any_end_date": int(no_end_date.sum()),
        },
        "rows_kept": len(kept),
        "covid_terminated": {
            "flagged": int(covid.sum()),
            "flagged_but_ended_before_2020": int((covid & ended_before_covid).sum()),
            "how": "label = TERMINATED and why_stopped matches the COVID keyword pattern "
                   "(why_stopped is used only for this flag, never as a feature)",
        },
    }


def split_balance(table: pd.DataFrame) -> list[dict]:
    """Rows and class balance for every split and both dataset versions."""
    rows = []
    for part, (first, last) in SPLIT_YEARS.items():
        for version in VERSIONS:
            subset = select_rows(table, (part,), version, final_evaluation=True)  # counting only
            terminated = int(subset["label"].sum())
            rows.append({
                "split": part, "years": f"{first}" if first == last else f"{first}-{last}",
                "version": version, "rows": len(subset), "terminated": terminated,
                "terminated_pct": pct(terminated, len(subset)),
            })
    return rows


# --- per-feature checks ---------------------------------------------------------------

def encode_for_auc(values: pd.Series, labels: pd.Series, kind: str) -> pd.Series:
    """Turn one feature into numbers for an AUC check.

    Numbers: missing -> the median. Flags: 0/1. Categories: each value -> its terminated rate.
    """
    if kind == "number":
        return values.fillna(values.median())
    if kind == "flag":
        return values.astype(float)
    return values.map(labels.groupby(values).mean())


def single_feature_auc(values: pd.Series, labels: pd.Series, kind: str) -> float:
    """How well this one feature alone separates the classes: 0.5 = not at all, 1.0 = perfectly.

    Reported as max(AUC, 1 - AUC), so a feature that predicts "completed" counts too.
    """
    encoded = encode_for_auc(values, labels, kind)
    if encoded.nunique() < 2:
        return 0.5
    auc = roc_auc_score(labels, encoded)
    return round(max(auc, 1 - auc), 3)


def is_missing(values: pd.Series, kind: str) -> pd.Series:
    """Missing means NaN for numbers and "MISSING" for categories. Flags are never missing."""
    if kind == "number":
        return values.isna()
    if kind == "category":
        return values == MISSING
    return pd.Series(False, index=values.index)


def distribution(values: pd.Series, kind: str, edges: np.ndarray | None) -> pd.Series:
    """Share of rows per value (categories, flags) or per bin (numbers, with NaN as its own bin)."""
    if kind == "number":
        binned = pd.Series(np.digitize(values.fillna(0), edges), index=values.index)
        values = binned.where(values.notna(), -1)
    return values.astype(str).value_counts(normalize=True)


def drift(train: pd.Series, other: pd.Series, kind: str) -> float:
    """Total variation distance between two distributions: 0 = identical, 1 = no overlap.

    Numbers are binned at the training deciles first.
    """
    edges = np.unique(np.nanquantile(train.astype(float), np.linspace(0.1, 0.9, 9))) if kind == "number" else None
    p, q = distribution(train, kind, edges), distribution(other, kind, edges)
    p, q = p.align(q, fill_value=0)
    return round(0.5 * float((p - q).abs().sum()), 3)


def audit_feature(feature: Feature, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """All checks for one feature (training rows for label checks, validation for drift)."""
    values, labels = train[feature.name], train["label"]
    missing = is_missing(values, feature.kind)
    auc = single_feature_auc(values, labels, feature.kind)
    shift = drift(values, val[feature.name], feature.kind)
    known = {c.name: c.known_at_start for c in COLUMNS}
    reasons = []
    if auc >= AUDIT_AUC_REVIEW_THRESHOLD:
        reasons.append("high AUC")
    if shift >= AUDIT_DRIFT_REVIEW_THRESHOLD:
        reasons.append("high drift")
    return {
        "feature": feature.name, "group": feature.group, "kind": feature.kind,
        "sources": list(feature.sources),
        "sources_known_at_start": sorted({known[s] for s in feature.sources}),
        "meaning": feature.meaning,
        "auc_train": auc,
        "missing_pct_completed": pct(int(missing[labels == 0].sum()), int((labels == 0).sum())),
        "missing_pct_terminated": pct(int(missing[labels == 1].sum()), int((labels == 1).sum())),
        "drift_train_vs_val": shift,
        "review_reasons": reasons,
        "needs_review": bool(reasons),
        "review_note": MANUAL_REVIEW_NOTES.get(feature.name, ""),
    }


def measured_leaks(trials: pd.DataFrame, table: pd.DataFrame) -> list[dict]:
    """AUC of a few excluded leakage columns on the same training rows, to show why they are out."""
    train_ids = set(select_rows(table, ("train",), "main")["nct_id"])
    train = trials[trials["nct_id"].astype(str).isin(train_ids)]
    labels = train["label"].astype(int)
    return [{"column": name, "auc_train": single_feature_auc(compute(train), labels, "number")}
            for name, compute in MEASURED_LEAKS.items()]


# --- the whole audit --------------------------------------------------------------------

def build_audit(trials: pd.DataFrame, table: pd.DataFrame, censoring_counts: dict) -> dict:
    """Collect every number for the audit into one dict."""
    train = select_rows(table, ("train",), "main")
    val = select_rows(table, ("val",), "main")
    known = {c.name: c.known_at_start for c in COLUMNS}
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "auc_review_threshold": AUDIT_AUC_REVIEW_THRESHOLD,
        "drift_review_threshold": AUDIT_DRIFT_REVIEW_THRESHOLD,
        "split_years": {k: list(v) for k, v in SPLIT_YEARS.items()},
        "versions": VERSIONS,
        "cleaning": cleaning_summary(trials),
        "split_balance": split_balance(table),
        "censoring_counts_snapshot": censoring_counts.get("registry_snapshot"),
        "censoring": censoring_table(trials, censoring_counts),
        "features": [audit_feature(f, train, val) for f in FEATURES],
        "caution_features": feature_names(("caution",)),
        "excluded_columns": [{"column": c, "known_at_start": known[c], "reason": r}
                             for c, r in EXCLUDED_COLUMNS.items()],
        "measured_leaks": measured_leaks(trials, table),
    }


def audit_markdown(a: dict) -> str:
    """Render the audit dict as Markdown."""
    c = a["cleaning"]
    late, covid = c["late_registration"], c["covid_terminated"]
    flagged = [f["feature"] for f in a["features"] if f["needs_review"]]
    years = {k: (f"{v[0]}" if v[0] == v[1] else f"{v[0]}–{v[1]}") for k, v in a["split_years"].items()}
    lines = [
        "# Leakage audit – Phase 8 features",
        "",
        f"Generated by `python -m src.dataset.features` on {a['generated_at']}. "
        "Do not edit by hand; numbers are also in `LEAKAGE_AUDIT.json`.",
        "",
        "**Goal:** the model may only use information that is known when a trial *starts*. "
        "This file lists every column, says whether it is a feature, and shows the checks.",
        "",
        "## Summary",
        "",
        f"- {c['trials_in']} finished trials in; **{late['dropped']} dropped** as late registrations; "
        f"**{c['rows_kept']} kept**.",
        f"- {covid['flagged']} kept trials flagged as terminated because of COVID-19. The main dataset leaves "
        "them out; the `with_covid` version keeps them so Phase 9 can compare.",
        f"- {len(feature_names(('strict',)))} strict features (known at start) + "
        f"{len(a['caution_features'])} caution features (planned at start, may be edited later).",
        f"- Flagged for a manual check (single-feature AUC ≥ {a['auc_review_threshold']} or train→validation "
        f"drift ≥ {a['drift_review_threshold']}): {', '.join(f'`{n}`' for n in flagged) if flagged else 'none'}. "
        "See the Review column in section 4.",
        f"- Split by start year: train {years['train']}, validation {years['val']}, "
        f"**test {years['test']} (final evaluation only)**, recent {years['recent']} (reported separately).",
        "",
        "## 1. Rows removed or flagged",
        "",
        "| Step | Trials | Note |",
        "|---|---|---|",
        f"| Finished trials in `trials.parquet` | {c['trials_in']} | COMPLETED or TERMINATED, started "
        f"{a['split_years']['train'][0]}–{a['split_years']['recent'][1]} |",
        f"| − registered after primary completion | {late['dropped']} | {late['terminated_pct_among_dropped']}% "
        "of these are terminated (vs about 12% overall) |",
        f"| **Kept** | **{c['rows_kept']}** | {late['kept_without_any_end_date']} trials have no end date at "
        "all, so they could not be checked and were kept |",
        f"| Flagged `covid_terminated` (kept, not dropped) | {covid['flagged']} | "
        f"{covid['flagged_but_ended_before_2020']} of them have a primary completion date before 2020 "
        "(checked by hand, see below) |",
        "",
        f"**Why late registrations are dropped:** {late['reason']}.",
        "",
        f"**COVID flag:** {covid['how']}. {COVID_PRE_2020_CHECK}",
        "",
        "## 2. Time-based split",
        "",
        "Train on older trials and judge on newer ones, like real use: the model is built today "
        "and used on trials that start later. A random split would let the model learn from trials "
        "that started *after* the ones it is tested on.",
        "",
        "| Split | Start years | Version | Rows | Terminated | % terminated |",
        "|---|---|---|---|---|---|",
    ]
    for s in a["split_balance"]:
        name = f"**{s['split']}**" if s["split"] == "test" else s["split"]
        lines.append(f"| {name} | {s['years']} | {s['version']} | {s['rows']} | {s['terminated']} | "
                     f"{s['terminated_pct']}% |")
    lines += [
        "",
        "Rules:",
        "- **train**: fit models and all preprocessing (encoding, filling gaps, scaling).",
        "- **val**: compare models and tune settings.",
        "- **test**: used **once**, for the final evaluation in Phase 9, never for tuning. "
        "`load_features()` refuses to load it unless `final_evaluation=True`.",
        "- **recent**: reported on its own, with the censoring note below.",
        f"- Versions: `main` = {a['versions']['main']}; `with_covid` = {a['versions']['with_covid']}.",
        "",
        "## 3. Right-censoring check (why 2019–2020 are reported separately)",
        "",
        "A trial that started in 2020 and is still running has no label yet, so it is missing from the "
        "dataset. The long-running trials are the ones missing, so recent years over-represent short trials. "
        f"API counts from registry snapshot {a['censoring_counts_snapshot']} (`data/censoring_counts.json`).",
        "",
        "| Start year | All interventional trials | Still running | UNKNOWN status | Finished (in dataset) | "
        "Finished trials lasting > 5 years | % terminated, trials finished within 5 years |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in a["censoring"]:
        lines.append(f"| {r['start_year']} | {r['all_trials']} | {r['still_running_pct']}% | "
                     f"{r['unknown_status_pct']}% | {r['finished_pct']}% | {r['finished_lasting_over_5y_pct']}% | "
                     f"{r['terminated_pct_if_finished_within_5y']}% |")
    lines += [
        "",
        "Reading it: the share still running grows for recent start years, and finished trials get shorter. "
        "The last column compares years fairly (every year has had at least 5 years to finish). "
        "If it stays in a narrow band, censoring mainly changes *which* trials are present, not the label rate. "
        "This is why the main test set is 2018 and 2019–2020 are a separate \"recent\" set.",
        "",
        "## 4. Features",
        "",
        "Checks (training rows, main version): **AUC** = how well the feature *alone* separates "
        "completed from terminated (0.5 = no signal); **missing %** by class (a big difference can "
        "mean the field is filled in *because of* the outcome); **drift** = total variation distance between "
        "train and validation (0 = same distribution, 1 = no overlap).",
        "",
        "| Feature | Group | Kind | From column(s) | Dictionary says | AUC | Missing % (completed / terminated) "
        "| Drift train→val | Review |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in a["features"]:
        missing = ("—" if f["kind"] == "flag"
                   else f"{f['missing_pct_completed']}% / {f['missing_pct_terminated']}%")
        review = f"**{', '.join(f['review_reasons'])}**" if f["needs_review"] else ""
        if f["review_note"]:
            review = f"{review}: {f['review_note']}" if review else f["review_note"]
        lines.append(f"| `{f['feature']}` | {f['group']} | {f['kind']} | {', '.join(f['sources'])} | "
                     f"{', '.join(f['sources_known_at_start'])} | {f['auc_train']} | {missing} | "
                     f"{f['drift_train_vs_val']} | {review} |")
    lines += [
        "",
        "## 5. Caution features (list them in the model card)",
        "",
        "Planned when the trial starts, but the registry keeps only the latest version, so the value may "
        "have been edited while the trial ran. Phase 9 trains with and without them.",
        "",
        *[f"- `{name}`" for name in a["caution_features"]],
        "",
        "## 6. Excluded columns",
        "",
        "| Column | Dictionary says | Why it is not a feature |",
        "|---|---|---|",
        *[f"| `{e['column']}` | {e['known_at_start']} | {e['reason']} |" for e in a["excluded_columns"]],
        "",
        "## 7. Excluded leakage columns, measured anyway",
        "",
        "Same AUC check on the same training rows. This shows what the model would have learned "
        "from them. None of these is a feature.",
        "",
        "| Column | AUC |",
        "|---|---|",
        *[f"| {m['column']} | {m['auc_train']} |" for m in a["measured_leaks"]],
    ]
    return "\n".join(lines) + "\n"


def write_audit(trials: pd.DataFrame, table: pd.DataFrame) -> Path:
    """Write LEAKAGE_AUDIT.json and .md. Returns the Markdown path."""
    years = range(SPLIT_YEARS["train"][0], SPLIT_YEARS["recent"][1] + 1)
    audit = build_audit(trials, table, load_censoring_counts(years))
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    AUDIT_MD.write_text(audit_markdown(audit), encoding="utf-8")
    return AUDIT_MD
