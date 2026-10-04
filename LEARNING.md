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
