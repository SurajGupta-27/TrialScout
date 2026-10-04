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
- **Interactions API**: Google's current docs use `client.interactions.create(model=..., input=...)` and read the answer from `interaction.output_text`. The older `client.models.generate_content` still works, but we follow the current docs.

### Design decisions
- **Gemini 3.8 Flash**: listed as free tier on Google's pricing page, fast, and recommended by Google for new projects.
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
