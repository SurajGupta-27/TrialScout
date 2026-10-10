# TrialScout – Roadmap

Tick a box only when the phase is finished: all tests pass, it runs end-to-end,
LEARNING.md is updated and the work is committed. Rules are in `CLAUDE.md`.

## Done – Phases 1-6 (LLM search assistant)
- [x] Phase 1: Project setup, config, Gemini client
- [x] Phase 2: ClinicalTrials.gov API v2 search and clean records
- [x] Phase 3: Filter extraction with JSON schema, validation, rate-limit-safe retries
- [x] Phase 4: Grounded summary, NCT ID and detail validation, end-to-end pipeline
- [x] Phase 5: Streamlit UI, deployed on Streamlit Community Cloud
- [x] Phase 6: 30-question evaluation, README with measured results

## Stage A – My own ML model
- [x] Phase 7: Dataset builder: finished interventional trials (~2010-2020) from
      ClinicalTrials.gov; label COMPLETED = 0, TERMINATED = 1; exclude WITHDRAWN at first
- [ ] Phase 8: Features known at trial start, leakage audit, time-based split
- [ ] Phase 9: Train and evaluate (Logistic Regression baseline, Random Forest,
      XGBoost/LightGBM), calibration, saved model, MODEL_CARD.md
- [ ] Phase 10: Add risk prediction (Low/Medium/High + disclaimer) to the pipeline

## Stage B – Real architecture
- [ ] Phase 11: FastAPI backend (/health, /search, /predict) with tests
- [ ] Phase 12: React frontend (Vite)
- [ ] Phase 13: Docker + GitHub Actions CI

## Stage C – Better search
- [ ] Phase 14: Embedding re-ranking with measured before/after results

## Stage D – Launch
- [ ] Phase 15: Free deployment + anonymous usage tracking
- [ ] Phase 16: Final evaluation, README and LEARNING.md update
