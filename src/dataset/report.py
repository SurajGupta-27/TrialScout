"""Write the data report (row count, class balance, missing values) and the data dictionary.

Called by `python -m src.dataset.build`. Every number comes from the built
dataset or from counts the API reported during the download; nothing is typed by hand.

Outputs (small, committed to git):
    data/data_report.json     all numbers, machine-readable
    data/data_report.md       the same numbers as readable tables
    data/DATA_DICTIONARY.md   every column, generated from src/dataset/columns.py
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DATASET_END_YEAR, DATASET_START_YEAR, LABELS
from src.dataset.columns import COLUMNS

REPORT_JSON = DATA_DIR / "data_report.json"
REPORT_MD = DATA_DIR / "data_report.md"
DICTIONARY_MD = DATA_DIR / "DATA_DICTIONARY.md"

# Words that mark a trial stopped because of COVID-19. Tuned on the real why_stopped texts:
# "covid" but not "Covidien" (a device company); "pandemi" also catches "pandemia";
# SARS-CoV-2 is written many ways ("SARSCov2", "SARS-CoV2").
COVID_PATTERN = re.compile(r"covid(?!ien)|corona ?virus|sars[- ]?cov[- ]?2|pandemi", re.IGNORECASE)
# Earlier pandemics that would otherwise match "pandemic" (e.g. trials started in 2010).
NOT_COVID_PATTERN = re.compile(r"h1n1|swine", re.IGNORECASE)


def mentions_covid(texts: pd.Series) -> pd.Series:
    """True where a why_stopped text mentions COVID-19 (a keyword match, so approximate)."""
    texts = texts.fillna("")
    return texts.str.contains(COVID_PATTERN) & ~texts.str.contains(NOT_COVID_PATTERN)


def pct(part: int, whole: int) -> float:
    """Percentage rounded to 1 decimal; 0 if the whole is 0."""
    return round(100 * part / whole, 1) if whole else 0.0


def missing_values(df: pd.DataFrame) -> list[dict]:
    """Missing count per column. For list columns, an empty list counts as missing."""
    result = []
    for col in COLUMNS:
        series = df[col.name]
        if col.dtype.startswith("list"):
            n_missing = int(series.map(len).eq(0).sum())
        else:
            n_missing = int(series.isna().sum())
        result.append({"column": col.name, "missing": n_missing, "missing_pct": pct(n_missing, len(df))})
    return result


def by_start_year(df: pd.DataFrame) -> list[dict]:
    """Rows, terminated count/rate and COVID mentions for each start year."""
    covid = mentions_covid(df["why_stopped"])
    result = []
    for year, group in df.groupby("start_year"):
        terminated = int(group["label"].sum())
        result.append({
            "start_year": int(year),
            "rows": len(group),
            "terminated": terminated,
            "terminated_pct": pct(terminated, len(group)),
            "why_stopped_mentions_covid": int(covid[group.index].sum()),
        })
    return result


def build_report(df: pd.DataFrame, excluded: Counter, state: dict) -> dict:
    """Collect every number for the report into one dict."""
    years = state["years"]
    api_counts = state.get("api_counts", {})
    downloaded = sum(years[y]["studies"] for y in years)
    terminated = int(df["label"].sum())
    terminated_rows = df[df["label"] == 1]
    covid_mentions = int(mentions_covid(terminated_rows["why_stopped"]).sum())

    not_downloaded = {}
    if api_counts:
        not_downloaded = {
            "WITHDRAWN (excluded on purpose)": api_counts["WITHDRAWN"],
            "other statuses with no final outcome": (
                api_counts["all_statuses"] - api_counts["WITHDRAWN"] - api_counts["COMPLETED|TERMINATED"]
            ),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start_year_range": [DATASET_START_YEAR, DATASET_END_YEAR],
        "registry_snapshots": sorted({years[y]["snapshot"] for y in years}),
        "api_counts_in_window": api_counts,
        "not_downloaded": not_downloaded,
        "downloaded_studies": downloaded,
        "excluded_after_download": dict(excluded.most_common()),
        "rows": len(df),
        "columns": df.shape[1],
        "class_balance": {
            status: {"label": label, "rows": int((df["label"] == label).sum()),
                     "pct": pct(int((df["label"] == label).sum()), len(df))}
            for status, label in LABELS.items()
        },
        "terminated_why_stopped_filled": int(terminated_rows["why_stopped"].notna().sum()),
        "terminated_why_stopped_mentions_covid": covid_mentions,
        "terminated_total": terminated,
        "start_date_type": {(k if isinstance(k, str) else "missing"): int(v)
                            for k, v in df["start_date_type"].value_counts(dropna=False).items()},
        "by_start_year": by_start_year(df),
        "missing_values": missing_values(df),
    }


def report_markdown(r: dict) -> str:
    """Render the report dict as Markdown tables."""
    lines = [
        "# Data report – Phase 7 dataset",
        "",
        f"Generated by `python -m src.dataset.build` on {r['generated_at']}. "
        "Do not edit by hand; numbers are also in `data_report.json`.",
        "",
        f"- Trials: interventional, start date {r['start_year_range'][0]}–{r['start_year_range'][1]}, "
        "final status COMPLETED (label 0) or TERMINATED (label 1).",
        f"- Registry data snapshot(s): {', '.join(r['registry_snapshots'])}",
        "",
        "## Row count",
        "",
        "| Step | Studies |",
        "|---|---|",
    ]
    if r["api_counts_in_window"]:
        lines.append(f"| All interventional studies in the window (any status) | {r['api_counts_in_window']['all_statuses']} |")
        for reason, n in r["not_downloaded"].items():
            lines.append(f"| − not downloaded: {reason} | {n} |")
    lines.append(f"| Downloaded (COMPLETED or TERMINATED) | {r['downloaded_studies']} |")
    for reason, n in r["excluded_after_download"].items():
        lines.append(f"| − excluded: {reason} | {n} |")
    if not r["excluded_after_download"]:
        lines.append("| − excluded after download (duplicates, bad dates, wrong status or type) | 0 |")
    lines += [f"| **Final dataset** | **{r['rows']}** rows × {r['columns']} columns |", ""]

    lines += ["## Class balance", "", "| Status | Label | Rows | % |", "|---|---|---|---|"]
    for status, c in r["class_balance"].items():
        lines.append(f"| {status} | {c['label']} | {c['rows']} | {c['pct']}% |")
    lines += [
        "",
        f"Of {r['terminated_total']} terminated trials, {r['terminated_why_stopped_filled']} give a "
        f"`why_stopped` reason and {r['terminated_why_stopped_mentions_covid']} of those mention COVID-19 "
        "(keyword match: covid, coronavirus, sars-cov-2, pandemic; H1N1 and \"Covidien\" excluded). "
        "Trials of every start year were stopped by COVID, not only those that started in 2020.",
        "",
        "## By start year",
        "",
        "| Start year | Rows | Terminated | % terminated | why_stopped mentions COVID |",
        "|---|---|---|---|---|",
    ]
    for y in r["by_start_year"]:
        lines.append(f"| {y['start_year']} | {y['rows']} | {y['terminated']} | {y['terminated_pct']}% | "
                     f"{y['why_stopped_mentions_covid']} |")

    lines += ["", "## Start date type", "", "| Type | Rows |", "|---|---|"]
    for k, v in r["start_date_type"].items():
        lines.append(f"| {k} | {v} |")

    lines += ["", "## Missing values", "",
              "List columns count an empty list as missing.", "",
              "| Column | Missing | % |", "|---|---|---|"]
    for m in r["missing_values"]:
        lines.append(f"| {m['column']} | {m['missing']} | {m['missing_pct']}% |")
    return "\n".join(lines) + "\n"


def write_report(df: pd.DataFrame, excluded: Counter, state: dict) -> Path:
    """Write data_report.json and data_report.md. Returns the Markdown path."""
    report = build_report(df, excluded, state)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(report_markdown(report), encoding="utf-8")
    return REPORT_MD


def data_dictionary_markdown() -> str:
    """Render the column list from columns.py as a Markdown table."""
    lines = [
        "# Data dictionary – `data/processed/trials.parquet`",
        "",
        "Generated from `src/dataset/columns.py` by `python -m src.dataset.build`. One row per trial.",
        "",
        "**Known at start** says whether a column could be used to predict the outcome "
        "*when the trial begins*:",
        "- **yes**: decided when the trial is designed or registered.",
        "- **caution**: planned at start but often edited later. The registry keeps only the "
        "latest version, so the value may reflect what happened (e.g. enrollment becomes the actual number).",
        "- **no**: only known during or after the trial. **Leakage: never use as a feature.**",
        "- **label**: the target.",
        "",
        "Note: every value comes from the *current* registry record, not a copy from the start date. "
        "Phase 8 audits this before choosing features.",
        "",
        "| Column | Type | Known at start | Meaning | API source |",
        "|---|---|---|---|---|",
    ]
    for c in COLUMNS:
        lines.append(f"| `{c.name}` | {c.dtype} | {c.known_at_start} | {c.meaning} | `{c.source}` |")
    return "\n".join(lines) + "\n"


def write_data_dictionary() -> Path:
    """Write data/DATA_DICTIONARY.md. Returns its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DICTIONARY_MD.write_text(data_dictionary_markdown(), encoding="utf-8")
    return DICTIONARY_MD
