"""Download finished interventional trials page by page, cache them, and resume after a stop.

Run from the repo root:
    python -m src.dataset.download                     # everything (~20 minutes)
    python -m src.dataset.download --max-pages 2       # smoke test: at most 2 pages per year
    python -m src.dataset.download --years 2010 2011   # only some years

Files on disk (git-ignored):
    data/raw/state.json                 progress for each year, so a stopped run can resume
    data/raw/2010/page_0001.json.gz     one cached API response per page
"""

import argparse
import gzip
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import requests

from src.config import (
    CT_API_BASE_URL,
    DATASET_END_YEAR,
    DATASET_FIELDS,
    DATASET_MAX_RETRIES,
    DATASET_PAGE_SIZE,
    DATASET_REQUEST_DELAY_SECONDS,
    DATASET_START_YEAR,
    DATASET_TIMEOUT_SECONDS,
    LABELS,
    RAW_DIR,
    USER_AGENT,
)

# https://clinicaltrials.gov/api/v2/version tells us when the registry data was last refreshed.
VERSION_URL = CT_API_BASE_URL.rsplit("/", 1)[0] + "/version"

# How often one year may be restarted (snapshot change or expired token) before giving up.
MAX_YEAR_RESTARTS = 3

# Types for the two network calls. They are passed in, so tests can use fakes.
Fetcher = Callable[[dict], dict]
SnapshotGetter = Callable[[], str]


class DownloadError(Exception):
    """Raised when the download cannot continue (API keeps failing, or counts don't add up)."""


class TokenExpiredError(DownloadError):
    """Raised when the API rejects a saved page token, so the year has to start again."""


# ---------------------------------------------------------------------------
# Building requests
# ---------------------------------------------------------------------------

def start_date_filter(year: int) -> str:
    """Essie expression for interventional studies that started in the given year."""
    return f"AREA[StudyType]INTERVENTIONAL AND AREA[StartDate]RANGE[{year}-01-01,{year}-12-31]"


def year_params(year: int, page_token: str | None, page_size: int = DATASET_PAGE_SIZE) -> dict:
    """Query parameters for one page of finished trials that started in `year`.

    The first page (no token) also asks for the total count, so we can check
    at the end that we received every study.
    """
    params = {
        "filter.overallStatus": "|".join(LABELS),  # COMPLETED|TERMINATED
        "filter.advanced": start_date_filter(year),
        "fields": DATASET_FIELDS,
        "pageSize": page_size,
    }
    if page_token:
        params["pageToken"] = page_token
    else:
        params["countTotal"] = "true"
    return params


# ---------------------------------------------------------------------------
# Talking to the API (real network calls)
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    """A session reuses the connection between pages and sends our User-Agent."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def backoff_seconds(attempt: int, retry_after: str | None) -> float:
    """How long to wait before retry number `attempt` (1, 2, ...).

    Uses the server's Retry-After header if it gives a number, otherwise
    5, 10, 20, 40, 60 seconds. Never more than 60 seconds.
    """
    if retry_after and retry_after.isdigit():
        return min(60.0, float(retry_after))
    return min(60.0, 5.0 * 2 ** (attempt - 1))


def get_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """GET a URL and return its JSON, retrying busy (429) and server (5xx) errors.

    Raises:
        TokenExpiredError: if a request with a pageToken gets HTTP 400.
        DownloadError: for other client errors, or when all retries fail.
    """
    params = params or {}
    error = "unknown error"
    for attempt in range(1, DATASET_MAX_RETRIES + 1):
        retry_after = None
        try:
            response = session.get(url, params=params, timeout=DATASET_TIMEOUT_SECONDS)
        except (requests.Timeout, requests.ConnectionError) as exc:
            error = f"network error ({type(exc).__name__})"
        else:
            code = response.status_code
            if code == 200:
                try:
                    return response.json()
                except requests.JSONDecodeError:
                    error = "invalid JSON in response"
            elif code == 429 or code >= 500:
                error = f"HTTP {code}"
                retry_after = response.headers.get("Retry-After")
            elif code == 400 and "pageToken" in params:
                raise TokenExpiredError(f"Page token rejected: {response.text[:200]}")
            else:
                raise DownloadError(f"HTTP {code} from ClinicalTrials.gov: {response.text[:200]}")

        if attempt < DATASET_MAX_RETRIES:
            wait = backoff_seconds(attempt, retry_after)
            print(f"    {error}; retry {attempt}/{DATASET_MAX_RETRIES - 1} in {wait:.0f}s")
            sleep(wait)
    raise DownloadError(f"Gave up after {DATASET_MAX_RETRIES} tries: {error}")


def make_fetcher(session: requests.Session) -> Fetcher:
    """Return a function that fetches one page of studies."""
    return lambda params: get_json(session, CT_API_BASE_URL, params)


def make_snapshot_getter(session: requests.Session) -> SnapshotGetter:
    """Return a function that reads the registry's current data timestamp."""
    return lambda: get_json(session, VERSION_URL)["dataTimestamp"]


def count_studies(fetch: Fetcher, status: str, year_from: int, year_to: int) -> int:
    """Count studies with a given status (or "" for any) in the start-year window."""
    params = {
        "filter.advanced": (
            f"AREA[StudyType]INTERVENTIONAL AND "
            f"AREA[StartDate]RANGE[{year_from}-01-01,{year_to}-12-31]"
        ),
        "countTotal": "true",
        "pageSize": 1,
        "fields": "NCTId",
    }
    if status:
        params["filter.overallStatus"] = status
    return int(fetch(params)["totalCount"])


# ---------------------------------------------------------------------------
# Files: cached pages and the resume state
# ---------------------------------------------------------------------------

def write_atomic(path: Path, data: bytes) -> None:
    """Write to a temp file, then rename. A crash mid-write never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_page(path: Path, data: dict) -> None:
    """Save one API response as gzipped UTF-8 JSON."""
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    write_atomic(path, gzip.compress(raw))


def load_page(path: Path) -> dict:
    """Read one cached API response."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_state(raw_dir: Path) -> dict:
    """Read download progress, or start fresh if there is none yet."""
    path = raw_dir / "state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"years": {}, "api_counts": {}}


def save_state(raw_dir: Path, state: dict) -> None:
    """Save download progress (atomically, so it is never half-written)."""
    text = json.dumps(state, indent=2, ensure_ascii=False)
    write_atomic(raw_dir / "state.json", text.encode("utf-8"))


def new_year_entry(snapshot: str) -> dict:
    """Progress record for a year that is starting from page 1."""
    return {
        "snapshot": snapshot,  # registry data timestamp when this year started
        "expected": None,  # totalCount reported by the API on page 1
        "pages": 0,
        "studies": 0,
        "next_token": None,
        "done": False,
    }


# ---------------------------------------------------------------------------
# The download loop
# ---------------------------------------------------------------------------

def download_year(
    year: int,
    state: dict,
    raw_dir: Path,
    fetch: Fetcher,
    get_snapshot: SnapshotGetter,
    max_pages: int | None = None,
    delay: float = DATASET_REQUEST_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Download (or resume) every page for one start year. Returns True if the year is complete.

    Restarts the year from page 1 if the registry data was refreshed since the
    year started (so one year never mixes two snapshots) or if the saved page
    token has expired.

    Raises:
        DownloadError: if the API keeps failing or the study count doesn't match.
    """
    key = str(year)
    year_dir = raw_dir / key

    for _ in range(MAX_YEAR_RESTARTS + 1):
        entry = state["years"].get(key)
        if entry and entry["done"]:
            print(f"{year}: already complete ({entry['studies']} studies), skipping")
            return True

        snapshot = get_snapshot()
        if entry and entry["snapshot"] != snapshot:
            print(f"{year}: registry data changed since this year started; restarting the year")
            entry = None
        if entry is None:
            shutil.rmtree(year_dir, ignore_errors=True)
            entry = new_year_entry(snapshot)
            state["years"][key] = entry
            save_state(raw_dir, state)
        elif entry["pages"]:
            print(f"{year}: resuming after page {entry['pages']}")

        try:
            finished = fetch_remaining_pages(year, entry, state, raw_dir, fetch, max_pages, delay, sleep)
        except TokenExpiredError as exc:
            print(f"{year}: {exc}; restarting the year")
            state["years"].pop(key)
            continue
        if not finished:
            return False

        # Same snapshot at the end as at the start means nothing changed in between.
        if get_snapshot() != entry["snapshot"]:
            print(f"{year}: registry data changed during download; restarting the year")
            state["years"].pop(key)
            continue
        if entry["studies"] != entry["expected"]:
            raise DownloadError(
                f"{year}: received {entry['studies']} studies but the API reported {entry['expected']}"
            )
        entry["done"] = True
        save_state(raw_dir, state)
        print(f"{year}: complete, {entry['studies']} studies in {entry['pages']} pages")
        return True

    raise DownloadError(f"{year}: restarted {MAX_YEAR_RESTARTS} times; try again later")


def fetch_remaining_pages(
    year: int,
    entry: dict,
    state: dict,
    raw_dir: Path,
    fetch: Fetcher,
    max_pages: int | None,
    delay: float,
    sleep: Callable[[float], None],
) -> bool:
    """Fetch pages until the API has no next page. Returns False if stopped by max_pages.

    After every page, the page file is saved first and then the state. If the
    program stops between the two, the page is simply fetched and saved again.
    """
    year_dir = raw_dir / str(year)
    pages_this_run = 0
    while entry["pages"] == 0 or entry["next_token"]:
        if max_pages is not None and pages_this_run >= max_pages:
            print(f"{year}: stopped after {pages_this_run} page(s) (--max-pages); run again to resume")
            return False
        if pages_this_run or entry["pages"]:
            sleep(delay)  # polite pause between requests

        data = fetch(year_params(year, entry["next_token"]))
        page_no = entry["pages"] + 1
        if page_no == 1:
            entry["expected"] = data.get("totalCount")

        save_page(year_dir / f"page_{page_no:04d}.json.gz", data)
        entry["pages"] = page_no
        entry["studies"] += len(data.get("studies", []))
        entry["next_token"] = data.get("nextPageToken")
        save_state(raw_dir, state)
        pages_this_run += 1
        print(f"{year}: page {page_no} saved ({entry['studies']}/{entry['expected']} studies)")
    return True


def record_excluded_counts(state: dict, raw_dir: Path, fetch: Fetcher) -> None:
    """Save how many studies in the window are WITHDRAWN or have another non-final status.

    These are not downloaded, but the data report shows them so the exclusions
    are measured, not guessed.
    """
    if state["api_counts"]:
        return
    a, b = DATASET_START_YEAR, DATASET_END_YEAR
    state["api_counts"] = {
        "all_statuses": count_studies(fetch, "", a, b),
        "WITHDRAWN": count_studies(fetch, "WITHDRAWN", a, b),
        "COMPLETED|TERMINATED": count_studies(fetch, "|".join(LABELS), a, b),
    }
    save_state(raw_dir, state)


def download_all(
    years: list[int],
    raw_dir: Path,
    fetch: Fetcher,
    get_snapshot: SnapshotGetter,
    max_pages: int | None = None,
    delay: float = DATASET_REQUEST_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Download every requested year, resuming from state.json. Returns the final state."""
    state = load_state(raw_dir)
    record_excluded_counts(state, raw_dir, fetch)
    for year in years:
        download_year(year, state, raw_dir, fetch, get_snapshot, max_pages, delay, sleep)
    return state


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Download finished trials from ClinicalTrials.gov.")
    parser.add_argument("--years", type=int, nargs="+",
                        default=list(range(DATASET_START_YEAR, DATASET_END_YEAR + 1)))
    parser.add_argument("--max-pages", type=int, default=None,
                        help="stop each year after this many new pages (for quick tests)")
    args = parser.parse_args()

    session = make_session()
    start = time.perf_counter()
    try:
        state = download_all(args.years, RAW_DIR, make_fetcher(session), make_snapshot_getter(session),
                             args.max_pages)
    except KeyboardInterrupt:
        print("\nStopped by user. Run the same command again to resume.")
        return
    except DownloadError as err:
        print(f"\nDownload failed: {err}\nRun the same command again to resume.")
        raise SystemExit(1)

    done = [y for y in args.years if state["years"].get(str(y), {}).get("done")]
    cached = sum(state["years"].get(str(y), {}).get("studies", 0) for y in args.years)
    print(f"\n{len(done)}/{len(args.years)} years complete, {cached} studies cached "
          f"in {RAW_DIR} ({time.perf_counter() - start:.0f}s this run)")


if __name__ == "__main__":
    main()
