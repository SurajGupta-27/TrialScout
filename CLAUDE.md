# TrialScout – Project Rules

These rules apply to every session. The plan and progress live in `ROADMAP.md`.

## Project context
TrialScout is my portfolio project (I'm a final-year B.Tech IT student). I must
understand and explain every part in interviews, so clarity beats cleverness.

Already built (Phases 1-6): Gemini filter extraction with JSON schema and
validation, ClinicalTrials.gov API v2 search, grounded summary with NCT ID and
detail validation, rate-limit-safe Gemini calls, Streamlit UI deployed on
Streamlit Community Cloud, tests (validation 6/6, extraction 15/15, 30-question
evaluation), README and LEARNING.md.

## Hard rule: 100% free
- No paid APIs, hosting or databases. No credit card anywhere.
- Gemini free tier only. Embeddings via a small local open-source model.
- Free hosting tiers only; check current limits before choosing.
- Open-source libraries only. If anything would cost money, stop and ask.

## Quality rules
- Take time; correctness over speed. One phase at a time.
- End of each phase: run ALL tests, run the feature end-to-end, fix every
  error, then STOP and give me:
  (a) how to run/test it on Windows PowerShell,
  (b) what you built and why in simple language,
  (c) known limitations or doubts,
  (d) wait for "next".
- Ask me when a decision is unclear instead of guessing.
- Never invent numbers; all metrics come from real runs saved to files.
- Small typed functions with docstrings; clean error handling.
- Never commit secrets or large data files.
- Keep everything that already works working.
- After each phase: update LEARNING.md (concepts, decisions, 3 interview Q&As),
  tick the phase in ROADMAP.md, and suggest a git commit message.

## How to run (Windows PowerShell, from the repo root)
```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py                                        # web UI
python -m src.pipeline "Phase 3 diabetes trials recruiting in India"   # terminal
python -m tests.test_validation                             # validator tests, no API key needed
python -m tests.try_extraction                              # 15-question filter check (uses Gemini)
python -m tests.evaluate                                    # 30-question evaluation (~5 min, uses Gemini)
python -m tests.test_dataset                                # dataset tests, offline
python -m src.dataset.download                              # download/resume trials (~20 min, ~460 MB)
python -m src.dataset.build                                 # build trials.parquet + reports (~1 min)
```
`tests.evaluate` overwrites `tests/eval_results.json` (the source of README numbers);
back it up first if you only want a check run.
The Gemini free tier allows about 15 requests per minute. Tests that call Gemini use up quota,
so run them on purpose, not in a loop.

## Code conventions already used (follow them)
- Settings and constants live only in `src/config.py` (model name, URLs, limits, allowed values).
- Each module wraps outside errors in one custom exception (`LLMError`, `TrialsAPIError`);
  `run_pipeline()` turns expected errors into a readable `result.error` instead of crashing.
- Raw API data is cleaned once at the boundary (`clean_study()`); the rest of the code uses simple dicts.
- `app.py` is a thin UI layer with no business logic.
- No LangChain or agent frameworks: plain Python that is easy to explain.
- Secrets: `.env` locally, `st.secrets` on Streamlit Cloud, read only through `get_api_key()`.
