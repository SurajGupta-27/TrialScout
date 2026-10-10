# TrialScout – Learning Notes

One section per build phase: the concepts used, why things were built this way, and likely interview questions.

---

## Phase 1 – Project setup

### What was built
| File | Purpose |
|---|---|
| `requirements.txt` | The four libraries the project depends on, so anyone can install them with one command. |
| `.gitignore` | Tells git which files to never commit, most importantly `.env` (holds the API key). |
| `.env.example` | A template showing *which* secrets are needed, without the real values. Safe to commit. |
| `src/config.py` | One place for settings: model name, API URL, timeouts, and `get_api_key()`. |
| `src/llm.py` | The Gemini client plus `generate_text()`, a small wrapper used by later phases. |

### Concepts
- **Virtual environment (`.venv`)**: an isolated folder of Python packages for this project only, so versions don't clash with other projects.
- **Environment variables and `.env`**: secrets are kept outside the code. `python-dotenv` loads `.env` into `os.environ` at startup. On Streamlit Cloud there is no `.env`, so `get_api_key()` falls back to `st.secrets`.
- **Single source of configuration**: `MODEL_NAME` is defined once in `config.py`. Switching models is a one-line change.
- **Wrapping errors**: the Gemini SDK can raise many exception types (invalid key, rate limit, network). `generate_text()` converts all of them into one `LLMError`, so the rest of the app only needs to handle one error type.
- **Caching the client**: `@lru_cache(maxsize=1)` on `get_client()` means the client is built once and reused instead of being recreated on every call.
- **`generate_content` API**: we call `client.models.generate_content(model=..., contents=..., config=...)` and read `response.text`. (Phase 1 first used the newer `client.interactions.create`, but Phase 3 testing showed that path can hang for hours on a rate-limit error. See the Phase 3 notes.)

### Design decisions
- **Model**: started with `gemini-3.8-flash`, but its free tier allows only 20 requests per day, which was used up during testing. Switched to `gemini-3.5-flash-lite`, also free tier and recommended by Google for new projects, with a higher allowance and about 1 s responses. Because the name lives only in `config.py`, the switch was a one-line change.
- **No framework (no LangChain)**: a direct SDK call is a few lines and every step is visible and easy to explain.
- **`src/` package**: keeps logic separate from the UI (`app.py`), so the same pipeline can run from the terminal, tests, or Streamlit.

### Likely interview questions
**Q: How do you keep your API key secure?**
A: It is never in the code. Locally it lives in `.env`, which is listed in `.gitignore` before the first commit, so it never reaches GitHub. In deployment it is stored in Streamlit's secrets manager. `.env.example` documents the variable name without the value.

**Q: Why put the model name in a config file?**
A: Models get updated and deprecated often. With one constant, changing the model is a one-line edit, and there is no risk of two parts of the app using different models by accident.

**Q: Why wrap SDK exceptions in your own `LLMError`?**
A: It decouples the app from the SDK. If the SDK changes its exception classes, only `llm.py` changes. Callers like the pipeline and UI catch one error type and show a friendly message.

---

## Phase 2 – Fetching trial data

### What was built
`src/trials_api.py` with three small functions:
| Function | Job |
|---|---|
| `build_params()` | Checks the filters and turns them into ClinicalTrials.gov query parameters. |
| `clean_study()` | Flattens one deeply nested API study into a simple dict. |
| `search_trials()` | Makes the HTTP request, handles errors, returns `(trials, total_count)`. |

Each clean record has: `nct_id`, `title`, `phase` (list), `status`, `sponsor`, `conditions`, `locations` ("City, Country", no duplicates), `countries`, `url`.

### Concepts
- **REST API + query parameters**: `requests.get(url, params={...})` builds the URL for us and encodes spaces and special characters safely.
- **How each filter maps to the API** (checked against the official OpenAPI spec at `/api/oas/v2`):
  | Our filter | API parameter | Example |
  |---|---|---|
  | condition | `query.cond` | `diabetes` |
  | status | `filter.overallStatus` | `RECRUITING` |
  | phase | `filter.advanced` | `AREA[Phase]PHASE3` |
  | country | `filter.advanced` | `AREA[LocationCountry]India` |
  Phase and country have no simple parameter, so we use the "advanced" filter written in Essie syntax, joining both parts with `AND`.
- **`query.*` vs `filter.*`**: `query.cond` is a ranked text search, so it also matches synonyms and related terms. `filter.*` is an exact yes/no filter.
- **`fields` parameter**: we ask only for the 8 fields we use instead of the full study record, which makes responses much smaller.
- **Timeouts**: every request has `timeout=30`. Without one, a slow server could freeze the app forever.
- **Exception order matters**: `requests.JSONDecodeError` is also a kind of `RequestException`, so it has to be caught first, or the user would get the wrong error message.

### Design decisions
- **Validate before calling the API**: phase and status are checked against the allowed values from the spec, so a typo gives a clear error instead of an HTTP 400.
- **Refuse searches with no filters**: with no filters the API would return every study it has, which is never what the user means.
- **Phase is a list**: some trials are "Phase 2/Phase 3", and the API returns `["PHASE2", "PHASE3"]`. Keeping the list avoids losing information.
- **Return the total count too**: we fetch 20 trials but report the real number of matches (for example "2444 total"), which is more honest in the UI.
- **Clean data at the boundary**: the rest of the app never sees the nested API format. If the API changes, only `clean_study()` needs updating.

### Likely interview questions
**Q: How do you filter by phase when the API has no phase parameter?**
A: With `filter.advanced=AREA[Phase]PHASE3`, the API's Essie expression syntax. I found it in the official OpenAPI spec, which also lists the allowed phase values. Country works the same way with `AREA[LocationCountry]`.

**Q: What happens if ClinicalTrials.gov is down or slow?**
A: Every request has a 30-second timeout. Timeouts, HTTP errors, connection errors and bad JSON are each caught and re-raised as one `TrialsAPIError` with a readable message, so the UI can show it instead of crashing.

**Q: Why clean the data instead of passing the raw JSON along?**
A: The raw response is deeply nested and has many sites per trial. Flattening it once keeps the rest of the code simple. It also means the LLM later gets a short, predictable input, which saves tokens and lowers the chance of mistakes.

---

## Phase 3 – Prompt engineering: question → filters

### What was built
In `src/llm.py`:
| Piece | Job |
|---|---|
| `FILTER_SYSTEM_PROMPT` | Instructions that tell Gemini how to turn a question into filters. |
| `FILTER_SCHEMA` | A JSON Schema, so Gemini's reply always has exactly the four keys. |
| `parse_json()` | Turns the reply text into a dict, and raises `LLMError` if it isn't valid JSON. |
| `validate_filters()` | Re-checks every value in Python and drops anything that isn't allowed. |
| `extract_filters()` | Connects these steps and returns `(filters, warnings)`. |

`tests/try_extraction.py` runs 15 varied questions and compares the answers with expected filters.

### How the prompt is structured
1. **Role and output format**: "You convert a question into search filters... Reply with JSON only."
2. **Field definitions**: each field, its allowed values, and plain-English mappings ("enrolling" → `RECRUITING`, "Phase III" → `PHASE3`, a city → its country).
3. **Rules for hard cases**: use null when unsure, null when the user names two phases or two countries, null for vague words like "late-stage", and all nulls for questions not about trials.
4. **Few-shot examples**: three examples (a full question, a partial one, and an off-topic one) show the exact output shape. Examples teach the format better than descriptions alone.

### Three layers of defence against bad LLM output
1. **Structured output**: `response_json_schema` makes Gemini produce JSON with those keys, and `enum` limits phase and status to allowed values.
2. **`parse_json()`**: if the reply still isn't a JSON object, we raise a clear `LLMError` instead of crashing later.
3. **`validate_filters()`**: Python checks again. It ignores unknown keys, turns `"null"` or `""` into `None`, normalises case (`"phase 3"`), and drops any phase or status that isn't allowed, adding a warning.
The idea is to never fully trust model output, even with a schema.

### Data-driven prompt change
The first prompt said "expand abbreviations" (HIV → "human immunodeficiency virus infection"). Testing against the real API showed this hurts:
| Search term | Recruiting trials found |
|---|---|
| `HIV` | 587 |
| expanded name | 198 |
ClinicalTrials.gov already matches synonyms, so the prompt now says "keep the user's own term".

### Reliability problems found while testing (and fixes)
- **Free-tier limits**: `gemini-3.8-flash` allows 20 requests per day, and `gemini-3.5-flash-lite` allows 15 per minute. A 429 error means "too many requests".
- **Hanging calls**: the SDK's `interactions` client retries 429 errors automatically and obeys the server's `Retry-After` wait, which was about 14 hours for a used-up daily quota. The program looked frozen. We moved to `generate_content`, which respects `HttpRetryOptions(attempts=1)` (no SDK retries), and set a 30 s timeout.
- **Our own retry** in `generate_text()` retries only "busy" errors (429, 500, 503), up to 3 tries. It waits as long as Google suggests ("retry in 28 s"), but gives up immediately if that is over 60 s, which means the daily quota is used up.
- **Dropped connections (fixed after Phase 7)**: a single `Server disconnected without sending a response` failed the whole request. Because the SDK's own retries are off, this network error reached us as a raw `httpx` exception, and our code treated every non-API error as fatal. Now `generate_text()` also retries network-level errors (`httpx.NetworkError`, `RemoteProtocolError`, `TimeoutException`, plus Python's `ConnectionError`/`TimeoutError`) with the same limit (3 tries) and waits (2 s, 4 s). If all tries fail, the `LLMError` says how many attempts were made and what the network error was. `tests/test_llm_retry.py` checks this with a fake client (no API key): one drop then success, every try dropping, and that a long rate-limit wait still fails fast.

### Measured result
After the fixes, `python -m tests.try_extraction` scored **15/15 fully correct**. Typical latency was about 1.2 s per question, and one question took 33.7 s because it waited out the per-minute limit. (This is a manual check. The proper 30-question evaluation is in Phase 6.)

### Likely interview questions
**Q: How do you make sure the LLM returns valid JSON?**
A: In three layers. First, Gemini's structured-output mode with a JSON Schema, using enums for phase and status. Second, a parse step that raises a clear error if the reply still isn't a JSON object. Third, Python validation that drops unknown keys and values outside the allowed lists, adding a warning. I never fully trust model output.

**Q: How does your prompt handle ambiguous questions?**
A: The rule is "null if not clearly mentioned, never guess." Vague terms like "late-stage" don't map to a phase. Two phases or two countries also give null, because the search supports one value. Off-topic questions give all nulls, and the pipeline then refuses to search. I tested each of these cases.

**Q: Tell me about a bug you found while testing.**
A: Calls started hanging for minutes. I called the REST API directly with curl and got HTTP 429 with a 14-hour retry delay, because the free daily quota was used up. A stack dump showed the SDK was quietly sleeping inside its own retry loop. I switched to the SDK path that lets me turn off built-in retries, then wrote my own retry. It waits only for short, per-minute limits and fails fast with a clear message otherwise.

---

## Phase 4 – Summary, validation, and the full pipeline

### What was built
| File | Job |
|---|---|
| `src/llm.py` → `summarise_trials()` | Sends the top 5 trials, as short labelled text, to Gemini with a strict "use only this data" prompt. |
| `src/validation.py` | Checks the summary line by line against the fetched trials. |
| `src/pipeline.py` | `run_pipeline(question)` runs all four steps and returns one `PipelineResult`. |
| `src/trials_api.py` → `format_phase()`, `format_status()` | Turn `PHASE2`+`PHASE3` into "Phase 2/Phase 3". Shared by the summary, the validator and the UI. |
| `tests/test_validation.py` | Feeds the validator summaries with deliberate mistakes and checks it catches each one. No API key needed. |

### The pipeline
```
question → extract_filters (LLM) → search_trials (API) → summarise_trials (LLM) → validate_summary (Python)
               ↓ all null?              ↓ 0 results?
         "not a trial question"     "no trials match", with no AI answer
```
Each step can fail on its own. The pipeline catches the expected errors (`LLMError`, `TrialsAPIError`, `ValueError`) and puts a readable message in `result.error`. The UI therefore never crashes, and if only the summary fails it still has the trials to show.

### How validation works
1. **Find every NCT ID** in each line with the regex `NCT\d{8}` (case-insensitive, so "nct0..." can't slip through).
2. **Unknown ID → remove the line.** If an ID is not among the fetched trials, the model invented it, so that line is deleted and listed in `removed_lines`.
3. **Known ID → check details.** The prompt makes each bullet end with `Phase: … . Status: … . Sponsor: … .`. This fixed format is what makes checking possible: regexes pull out each value, which is compared with the real record.
   - phase: compared as sets of codes, so "Phase 2/3" and "Phase 2/Phase 3" both match `{PHASE2, PHASE3}`.
   - status: compared on letters only, so "Active, not recruiting" matches "active not recruiting".
   - sponsor: passes if one name contains the other.
   A mismatch keeps the line but adds `[FLAGGED: …]` and records the exact difference.
4. `hallucination_count` = invented IDs + wrong details. Phase 6 reports it.

### Design decisions
- **Remove vs flag**: an invented trial is pure fiction, so it is removed. A real trial with one wrong detail is still useful, so it is flagged and the user is warned.
- **No AI answer when there are no results**: if the search returns nothing, we never call the summary model. It would have nothing real to summarise and could only invent.
- **Grounding the prompt**: the model sees only short labelled facts (ID, title, phase, status, sponsor, countries), so everything it needs is in front of it. The prompt forbids anything outside the data.
- **Structured output, then programmatic checks**: asking for a fixed bullet format is a prompt-engineering choice made so that the Python validator can check it.
- **Validate against all 20 fetched trials, not just the top 5**: the rule is "the ID must exist in the fetched results".

### Measured results
- `python -m tests.test_validation`: 5/5 tests pass. Planted invented IDs, wrong phase, wrong status, wrong sponsor and a lower-case ID are all caught.
- Live run, "Phase 3 diabetes trials recruiting in India": 15 trials found, 5 summarised, 0 problems caught, 4.6 s total (2.3 s extract, 0.7 s search, 1.6 s summary).
- No-results and off-topic questions return the right message, with no AI summary.

### Likely interview questions
**Q: How do you stop the LLM from hallucinating trials?**
A: Three ways. First, grounding: the model only sees the trials we fetched, with a "use only this data" instruction. Second, a fixed bullet format with labelled fields. Third, a Python validator that removes any line whose NCT ID wasn't fetched, and flags lines whose phase, status or sponsor don't match the record. If the search finds nothing, we don't call the LLM at all.

**Q: Why remove some lines but only flag others?**
A: An unknown NCT ID means the trial may not exist, and showing it could send a patient or researcher after a fake study, so it is removed. A real trial with a wrong detail is still a real lead, so it stays, clearly flagged, with the correct value shown.

**Q: How do you know your validator works if the model rarely hallucinates?**
A: I tested it with planted errors. `tests/test_validation.py` builds summaries with invented IDs and wrong phase, status and sponsor values, then asserts that each one is caught. That needs no API calls, so it is fast and gives the same result every run.

---

## Phase 5 – Streamlit UI and deployment

### What was built
`app.py`, a single page:
1. **Search box** inside an `st.form`, so pressing Enter or clicking Search runs one query.
2. **Example buttons**: each uses an `on_click` callback that writes the question into the search box's session state and sets a "run now" flag.
3. **Extracted filters** as four `st.metric` tiles. "Any" means the AI found nothing for that field, so the user can see exactly how the question was read.
4. **Results table** (`st.dataframe`) with a `LinkColumn`: each NCT ID opens its ClinicalTrials.gov page.
5. **Validated summary**, followed by a green "validation passed" box or a yellow list of removed and flagged items, plus a note when the summary skipped a trial.
6. **Timings** for each step, and a "not medical advice" disclaimer.

### Concepts
- **How Streamlit runs**: the whole script re-runs from top to bottom on every click. `st.session_state` is the dictionary that survives between re-runs. We use it for the search text, the "run now" flag and the results cache.
- **Widget callbacks**: a widget's value can only be changed before the widget is drawn. That is why the example buttons use `on_click`, which runs before the next re-run, instead of setting the value afterwards.
- **UI is a thin layer**: `app.py` has no business logic. It calls `run_pipeline()` and displays the result, so the same pipeline works from the terminal, tests and the web page.
- **Caching only successes**: answers are cached per session to save free-tier quota. Results with an error are not cached, so a temporary rate limit doesn't "stick" for an hour. That is why we wrote a small dict cache instead of `@st.cache_data`, which would cache errors too.
- **Headless UI testing**: `streamlit.testing.v1.AppTest` runs the app without a browser, clicks buttons and reads what was drawn. We used it to test the example buttons, the off-topic case and the empty-input case.

### Omission check (added in this phase)
In one UI test the model summarised only 3 of the 5 trials. That is not a hallucination, but the user should know. `validate_summary()` now also returns `missing_ids`, the trials the model was asked to cover but skipped, and the UI shows them. The prompt now says "one bullet for EVERY trial (do not skip any)". This is tested in `test_missing_trials_are_reported`.

### Deploying on Streamlit Community Cloud
1. Push the repo to GitHub, after checking that `.env` is not in `git status`.
2. Go to https://share.streamlit.io, sign in with GitHub, and click **Create app**.
3. Pick the repo `SurajGupta-27/TrialScout`, branch `main`, main file `app.py`.
4. Under **Advanced settings**, choose Python 3.12 and paste this into **Secrets** (TOML format):
   ```toml
   GEMINI_API_KEY = "your-real-key"
   ```
5. Click **Deploy**. Cloud installs `requirements.txt` and starts the app. `get_api_key()` finds no `.env` there and reads `st.secrets["GEMINI_API_KEY"]` instead.
6. To change the key later, open the app's **Settings → Secrets**, then reboot the app.

### Likely interview questions
**Q: How does Streamlit keep state if the script re-runs on every click?**
A: Through `st.session_state`, a per-user dictionary that survives re-runs. I store the current question, a flag set by the example buttons, and a cache of successful results there. Everything else is recomputed, which keeps the code simple.

**Q: How do you handle secrets differently locally and in production?**
A: The code has one function, `get_api_key()`. It checks environment variables first (filled from `.env` by python-dotenv locally), then `st.secrets`, which Streamlit Cloud fills from its encrypted secrets settings. The key is never in the repo, and `.streamlit/secrets.toml` is in `.gitignore` too.

**Q: Why not put the logic straight into `app.py`?**
A: Separation of concerns. The UI only displays a `PipelineResult`. The pipeline is plain Python I can run from the terminal and test without a browser. If I changed the UI framework, the pipeline wouldn't change at all.

---

## Phase 6 – Evaluation and README

### What was built
| File | Purpose |
|---|---|
| `tests/test_questions.json` | 30 new questions (not the 15 the prompt was tuned on) in 8 categories, each with expected filters. A free-text answer can list several acceptable values (e.g. `["CKD", "chronic kidney disease"]`). |
| `tests/evaluate.py` | Runs every question through the **full** pipeline, scores the filters, collects validation counts and timings, prints a report, and saves everything to `tests/eval_results.json`. |
| `README.md` | Overview, Mermaid architecture diagram, how validation works, measured results, setup, limitations. |

### Metrics and how they are measured
- **Filter extraction accuracy**: a question counts as correct only if *all 4* fields are right. Phase and status must match exactly. Condition and country pass on a case-insensitive "one contains the other" match, so "Alzheimer" and "Alzheimer's" both count. Per-field and per-category scores show *where* errors happen.
- **Hallucinations caught**: invented IDs removed plus wrong details flagged, taken from each `ValidationResult`. Omissions are counted separately, because a skipped trial is not a made-up one.
- **Response time**: wall-clock time for `run_pipeline()` per question, reported as mean, median and max. The pause between questions that keeps us under the free tier's 15 requests per minute is *not* counted.

### Results (2026-10-04, gemini-3.5-flash-lite)
| | Run 1 | Run 2 |
|---|---|---|
| All 4 fields correct | 28/30 | 30/30 |
| Hallucinations caught | 0 | 0 |
| NCT IDs verified | 121 | 131 |
| Mean response time | 4.74 s | 5.59 s |

- **Run 1 found a real bug**: typos ("diabetis") were kept, and the search returned 0 trials. "diabetis" finds 7 studies in the whole registry, against 24,469 for "diabetes".
- **The fix** makes the prompt always correct spelling. Its examples use different words than the test set ("asthama", "cancr").
- **Run 2 scored 30/30**, but because the fix came from this test set, run 2 is somewhat optimistic. Both runs are kept and reported.

### Concepts
- **Evaluation set vs tuning set**: if you tune a prompt on the same questions you score it on, the score goes up without proving the prompt generalises. That is why the 30 questions are new, why the typo fix uses different example words, and why the README says run 2 is optimistic.
- **Report the honest number**: "0 hallucinations caught" is not a failure of the validator. It means grounding worked on these 30 questions. The planted-error tests prove the validator would catch problems.
- **Keep raw results**: every README number comes from a saved JSON file with a timestamp, so anyone can check it.
- **Mermaid**: a text format for diagrams that GitHub renders automatically, so the architecture diagram lives in version control next to the code.

### Likely interview questions
**Q: How did you evaluate the system?**
A: With 30 hand-labelled questions across 8 categories, including tricky ones: typos, abbreviations, city to country, two phases, and off-topic. A script runs each one through the live pipeline. It reports exact-match filter accuracy, per-field accuracy, hallucinations caught by the validator, and response time. The first run scored 28/30, and both misses were typos that made the search return nothing. After a prompt fix it scored 30/30. I report both, because the fix was informed by the test set.

**Q: Your validator caught 0 hallucinations. Is it useless?**
A: No. Zero means the grounded prompt worked on these questions: all 131 IDs and their details were correct. But models change and inputs vary, so the validator is a safety net, like a unit test that usually passes. I proved it works with planted-error tests: invented IDs, wrong phase, status and sponsor, and skipped trials are all caught.

**Q: What would you improve next?**
A: A larger, unseen test set to get a fair score after tuning. Support for several phases or countries per search (the API supports `OR` in the advanced filter). Showing the city-level locations that matched the user's country. And caching across users, to stretch the free-tier quota on the deployed app.

---

## Phase 7 – Dataset builder (start of my own ML model)

### What was built
| File | Job |
|---|---|
| `src/dataset/download.py` | Downloads finished interventional trials (start date 2010–2020) page by page, caches every page, and resumes after a stop. |
| `src/dataset/columns.py` | One list describing every column: type, API source, meaning, and whether it is known at trial start. |
| `src/dataset/build.py` | Flattens cached studies into one row each, labels them, applies and counts exclusions, saves `data/processed/trials.parquet`. |
| `src/dataset/report.py` | Generates `data/data_report.md/.json` (row count, class balance, missing values) and `data/DATA_DICTIONARY.md`. |
| `tests/test_dataset.py` | 15 offline tests: parsing, labels, exclusions, report numbers, and download/resume with a fake API. |

### The label
Only trials with a **final** outcome are used: `COMPLETED = 0`, `TERMINATED = 1`. Everything else is left out:
- **WITHDRAWN** (7,352 in the window): these trials stopped *before enrolling anyone*. That is a different event from "started, then failed". Their actual enrollment is 0, so a model would learn "enrollment 0 → withdrawn" instead of anything useful.
- **RECRUITING, ACTIVE, SUSPENDED, UNKNOWN…** (44,395): we don't know how they end yet.

### Concepts
- **Pagination with tokens**: the API returns at most 1,000 studies per page plus a `nextPageToken`. Sending that token back gives the next page. There is no "page 7" jump, so the token is the bookmark we save.
- **Resumable downloads**: after each page we save the page file *first*, then `state.json` (pages done, next token). If the program stops between the two, that page is simply fetched again. Both files are written to a temp file and then renamed (`os.replace`), so a crash can never leave a half-written file.
- **Consistent snapshots**: the registry is refreshed daily. If a year's download straddled a refresh, it could mix two versions and miss or duplicate studies. We store the API's `dataTimestamp` when a year starts and check it again at the end (and when resuming). If it changed, that year is downloaded again.
- **Polite downloading**: 1 s pause between requests, a `User-Agent` naming the project, a 60 s timeout, and retries with exponential backoff (5, 10, 20, 40 s) only for 429/5xx/network errors. Other errors stop immediately with a clear message.
- **Dependency injection for testing**: `download_year()` receives the `fetch` and `get_snapshot` functions as arguments. The real program passes functions that call the API; the tests pass a `FakeAPI`, so resume, expired tokens and snapshot changes are tested in milliseconds without the network.
- **Verifying completeness**: page 1 asks for `countTotal`. At the end of each year the number of studies received must equal that count, or the download fails loudly. All 11 years matched exactly (155,288 studies).
- **Parquet vs CSV**: Parquet stores column types (dates, nullable integers, booleans) and list columns (conditions, countries) directly. It is also compressed: 19 MB instead of the 463 MB raw cache.
- **Nullable types**: pandas `Int64`/`boolean` (capital I) can hold missing values. Plain `int64` can't, and would silently turn a column with gaps into floats.
- **Leakage**: a column is leakage if it is only known *after* the outcome happens: completion date, `why_stopped`, `has_results`, *actual* enrollment. They are kept for analysis, but the data dictionary marks them `no`, so Phase 8 can't use them by accident.

### Design decisions
- **One query per start year**: about 15 pages each. Resuming, verifying counts and restarting after a refresh all work per year, so a problem costs at most one year, not the whole download. The 11 per-year counts were checked to add up exactly to the single-query total.
- **Skip `resultsSection`**: it is posted after the trial ends (pure leakage) and was over half of each page's size.
- **Keep raw pages, build separately**: Phase 8 may need fields we didn't flatten yet. Re-building takes about 50 s, while re-downloading takes about 20 min.
- **Generate docs from code**: the data dictionary is produced from `columns.py`, and the report from the data. Nothing is typed by hand, so the numbers can't drift.
- **Windows**: every file is opened with `encoding="utf-8"`. Without it, Python on Windows uses cp1252 and crashed on the first page during planning.
- **COVID keyword count, tuned on real text**: a first pattern also matched the 2009 H1N1 "pandemic" and the company "Covidien", and missed spellings like "SARSCov2". The final pattern was checked against the actual `why_stopped` texts.

### Measured results (registry snapshot 2026-10-09)
- 207,035 interventional trials started 2010–2020; 155,288 are COMPLETED or TERMINATED; 0 more were dropped after download (no duplicates, no missing start dates).
- **Class balance: 137,913 completed (88.8%) vs 17,375 terminated (11.2%)**, which is imbalanced. Accuracy will be a misleading metric in Phase 9 (always guessing "completed" scores 88.8%).
- Terminated rate per start year stays between 10.3% and 12.4%.
- 1,336 of the terminated trials mention COVID-19 in `why_stopped`. This happened in *every* start year, not just 2020, because long trials were still running in 2020.
- Real interruption test: the download was killed in the middle of 2011. The next run skipped 2010, resumed 2011 after page 3, and finished with exact counts. One real network error was retried and recovered automatically.

### Likely interview questions
**Q: How did you build your training dataset?**
A: From the ClinicalTrials.gov API v2. I downloaded every interventional trial that started 2010–2020 and ended as COMPLETED (label 0) or TERMINATED (label 1), which gave 155,288 trials with about 11% terminated. The download is split by start year, caches every page, and resumes from a saved page token. It checks that the number of studies received matches the API's total. A separate build step flattens each study into 48 columns, saves Parquet, and generates a data report and data dictionary.

**Q: Why did you exclude withdrawn trials?**
A: A withdrawn trial never enrolled anyone. That's a different question ("will this trial start?") from mine ("will a running trial finish?"). Its enrollment is also 0 by definition, so the model would learn a shortcut instead of a real pattern. I recorded how many I excluded (7,352) so the choice is visible, and I may model them separately later.

**Q: What is data leakage and how did you guard against it at this stage?**
A: Leakage is when a feature contains information you wouldn't have at prediction time, so the model looks great in testing and fails in real use. For example, `why_stopped` is filled almost only for terminated trials, so it nearly *is* the label. Every column in my data dictionary is marked yes, caution or no for "known at trial start". "Caution" covers fields like enrollment, which is planned at the start but overwritten with the actual number at the end. Phase 8 audits these before any feature is used.

---

## Phase 8 – Features, leakage audit, time-based split

### What was built
| File | Job |
|---|---|
| `src/dataset/filters.py` | Row checks: late registration, COVID-terminated flag. These read "after the fact" columns only to drop or flag rows. |
| `src/dataset/features.py` | The feature list (one `Feature` per column, grouped strict / caution), the reason for every excluded column, the feature computation, the split, and `load_features()`. |
| `src/dataset/audit.py` | Generates `data/LEAKAGE_AUDIT.md/.json`: rows removed, split balance, censoring check, and per-feature checks. |
| `tests/test_features.py` | 11 offline tests: no column forgotten, no leakage column used, filters, feature values, split years, test-set lock, AUC and drift checks. |

`python -m src.dataset.features` writes `data/processed/features.parquet` (130,253 rows, 58 features: 50 strict + 8 caution) and the audit.

### Concepts
- **Kinds of leakage found in this data:**
  - *Target leakage*: a column filled in because of the outcome (`why_stopped`, `has_results`).
  - *Overwritten values*: `enrollment_count` is planned at the start but replaced by the actual number at the end (98% of rows). Terminated trials enroll fewer people (median 19 vs 60), so on its own it scores AUC 0.741, far above any real feature (best: 0.584).
  - *Train/serve mismatch*: a live recruiting trial only has the *planned* enrollment, so a model trained on actual counts would be used on a different quantity.
  - *Selection leakage*: 25,035 trials were registered only after they ended, and only 1.9% of them are terminated. They are not real "at start" cases, so they are dropped.
- **Time-based split**: train 2010–2016, validation 2017, test 2018. A random split lets the model learn from trials that started *after* the ones it is judged on. That can't happen in real use, so a random split gives optimistic scores.
- **Never tune on the test set**: if you pick settings by looking at test scores, the test score is no longer an honest estimate. `load_features()` refuses to return the test split unless `final_evaluation=True`.
- **Right-censoring**: a trial that started in 2020 and is still running has no label yet, so it is missing from the data. The long trials are the ones missing (5.9% of finished 2019 trials and 2.6% of finished 2020 trials lasted more than 5 years, against 11.2% for 2010), so 2019–2020 are a separate "recent" set.
- **Single-feature AUC**: score each feature on its own. If one simple column almost predicts the label, it is often leakage. 0.5 means no signal; I report max(AUC, 1−AUC).
- **Drift (total variation distance)**: half the sum of the differences between two distributions. 0 means the same, 1 means no overlap. It found a registry rule change (below).
- **Preprocessing is fitted on train only**: encoding, filling gaps and scaling are learned from training rows in Phase 9. Fitting them on all years would let test-year information leak in.

### Design decisions (decided with the project owner)
- `enrollment_count` was reclassified from *caution* to *leakage* and excluded.
- Late registrations are dropped, and the count and reason are in the audit and the data report.
- COVID: `covid_terminated` is a flag (from `why_stopped`, data cleaning only). The `main` version leaves out the 1,285 flagged trials; `with_covid` keeps them for a Phase 9 comparison. I read the 69 flagged trials with all their dates before 2020. They are real COVID stops: for terminated trials the registry often records the last patient's date (before the pause) as the end date.
- *Caution* features (sites, countries, collaborators, outcome counts, eligibility text length) are separate, so Phase 9 can measure them, and the model card will list them.
- Every `trials.parquet` column is either a feature source or listed in `EXCLUDED_COLUMNS` with a reason. A test enforces this, so a new column can't slip in silently.
- The audit is generated from code, like the data report: no hand-typed numbers.

### Measured results (snapshot 2026-10-09)
- 155,288 → 130,253 trials after dropping 25,035 late registrations.
- Main version: train 77,902 (12.3% terminated), validation 13,259 (12.1%), test 13,299 (11.7%), recent 24,508 (11.6%).
- No feature reaches single-feature AUC 0.65 (highest: `eligibility_criteria_chars` 0.584, `phase` 0.578).
- **Dropped after review:** `is_fda_regulated_drug/device` drift 0.831 between train and validation. They are MISSING in 87.2% of training rows and 0.1% of test rows, because the fields became required for new registrations in 2017. The model would learn from "MISSING", a value live trials never have (train/serve mismatch), so both were moved to the excluded columns. Section 8 of the audit keeps measuring them.

### Likely interview questions
**Q: How did you make sure your model only uses information available at trial start?**
A: Every column is labelled yes / caution / no in the data dictionary, and a test checks that features only come from allowed columns. Every excluded column has a written reason. Then a generated audit scores each feature on its own; a suspiciously strong one would be flagged. As a contrast, I measured the excluded `enrollment_count`: AUC 0.741 alone, against 0.584 for the best real feature, because it stores the *actual* enrollment, which is low when a trial stops early.

**Q: Why a time-based split instead of a random one?**
A: In real use the model is trained on the past and used on trials starting later. A random split mixes years, so the model learns patterns from the "future". I train on 2010–2016, tune on 2017, and test once on 2018. 2019–2020 are kept separate because many of those trials are still running, which is right-censoring. The code refuses to load the test split unless it's the final evaluation.

**Q: What was the most surprising data problem?**
A: 16% of "finished" trials were registered only after they had ended, and 98% of those were completed: sponsors register finished trials so they can publish them. Kept in, the model would learn "late registration means success", which is useless for a new trial. Another: the drift check found that two FDA fields became required in 2017, so they are almost always missing in my training years and almost never missing afterwards.

---

## Phase 9 – Training and evaluation

*Status: model trained, tuned and saved; the one-time 2018 test is waiting for approval of the chosen model.*

### What was built
| File | Job |
|---|---|
| `src/model/pipeline.py` | One scikit-learn `Pipeline` per model: one-hot categories, median fill + `log1p` + scaling for counts, flags as 0/1, then the model. |
| `src/model/metrics.py` | PR-AUC, ROC-AUC, Brier, best-F1 threshold, confusion matrix, reliability table, risk bands. |
| `src/model/calibration.py` | Sigmoid / isotonic calibrators, out-of-fold comparison, and `CalibratedModel` (the saved object). |
| `src/model/train.py` | The grid on 2017 validation, model choice, calibration, bands, importance, saving, reports. |
| `src/model/final_test.py` | The one-time 2018 test: needs `--approved`, refuses a second run, checks the model file's sha256. |
| `src/model/predict.py` | `predict_risk()` → probability + Low/Medium/High, for Phase 10. |
| `src/model/report.py`, `card.py` | Generate `reports/model_validation.md` and `MODEL_CARD.md` from the result files. |
| `tests/test_model.py` | 10 offline tests on synthetic data. |

### Concepts
- **Dummy baseline**: always predicting "completed" scores 87.9% accuracy on 2017 but finds no terminated trial. That's why accuracy is useless here, and why every table shows the dummy.
- **PR-AUC vs ROC-AUC**: ROC-AUC asks "is a random terminated trial ranked above a random completed one?" (0.5 = guessing). PR-AUC focuses on the rare class: a no-skill model scores the base rate (0.1207 here), so "0.26" means a bit more than twice the baseline.
- **Pipeline fitted on train only**: the encoder, medians and scaler learn only from 2010–2016, so validation and test rows can't leak into preprocessing.
- **Calibration**: a model can rank well while its probabilities are off. Sigmoid (Platt) fits a 2-number logistic curve on the scores; isotonic fits a step function. I chose by 5-fold cross-validated Brier score *within* 2017, and all post-calibration validation numbers are out-of-fold, so no trial is scored by a calibrator fitted on it.
- **Threshold choice**: precision/recall depend on a cut-off. I picked the best-F1 threshold on validation and freeze it for the test.
- **Permutation importance**: shuffle one feature and measure how much validation PR-AUC drops. It works for any model and keeps one-hot pieces together.

### Measured results (validation 2017, run of 2026-10-10, 5.8 min)
- **Chosen: LightGBM** (300 trees, 31 leaves, min 200 samples per leaf), `main` data, strict+caution features, sigmoid calibration, **0.44 MB**.
- Validation: **PR-AUC 0.26** (dummy 0.1207), **ROC-AUC 0.73**, Brier 0.0982 (dummy 0.1207). At the best-F1 threshold 0.1575: precision 0.245, recall 0.575, F1 0.343.
- **Strict vs strict+caution**: best PR-AUC 0.230 vs 0.261. The caution features help, and three of them are among the five most important (`has_us_site`, `n_locations`, `eligibility_criteria_chars`).
- **main vs with_covid** training data: no real difference (best 0.2611 vs 0.2595 on the same validation rows).
- Random Forest strict+caution scored 0.2611, slightly above LightGBM's 0.2599, but its file was 27 MB, over the 25 MB budget. LightGBM is 60× smaller for the same score.
- Calibration barely mattered (Brier: none 0.09825, sigmoid 0.09818, isotonic 0.09858): LightGBM's raw scores were already well calibrated.
- **Risk bands** (base rate b = 0.1207): Low 59.8% of trials (5.9% actually terminated), Medium 29.4% (18.0%), High 10.7% (30.4%). High-band trials were terminated about 5× as often as Low-band ones.

### Likely interview questions
**Q: Your ROC-AUC is only 0.73. Is the model any good?**
A: It's moderate, and I expected that: termination often depends on things not in the registry at the start (funding, slow recruitment, safety findings). In Phase 8 no single start-time feature went above AUC 0.584. What matters is the honest comparison: PR-AUC 0.26 against 0.12 for the dummy, and the High band being terminated 30% of the time against 6% for Low. That's useful for ranking or flagging, not for deciding about one trial. I say this plainly in the model card.

**Q: How did you avoid overfitting to your test set?**
A: All choices (model, settings, feature set, data version, calibration, threshold, bands) were made on the 2017 validation year only. The 2018 test is run once, by a separate script that needs an explicit `--approved` flag, refuses to run twice, and checks the model file's hash, so the reported test score belongs to exactly the saved model.

**Q: Why LightGBM and not Random Forest?**
A: On validation they were tied (PR-AUC 0.2599 vs 0.2611), but the Random Forest file was 27 MB and over my 25 MB free-hosting budget, while LightGBM was 0.44 MB and fit 3× faster. The size budget was set before training and enforced in the selection code.
