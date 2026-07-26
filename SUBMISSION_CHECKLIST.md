# Submission Checklist

Status as of this pass. Re-verify anything below that changes after this
point — this file is a checklist, not a guarantee.

## Code and tests

- [x] Full test suite passing — 253/253, 0 failures
- [x] No secrets, API keys, or credentials in tracked files (scanned)
- [x] `.gitignore` excludes `data/` (268 MB, regenerable), `models/checkpoints/`
      (48 MB, not needed to run the demo), `.venv/`, `node_modules/`
- [x] Trained models committed (`models/risk_model.joblib`,
      `models/forecast_model.joblib`) so the app runs with no training step
- [x] No stray `__pycache__/`, `.DS_Store`, or editor backup files

## Documentation

- [x] `README.md` — business summary first, technical detail later
- [x] `PROJECT_LOG.md` — full design history, header summary current
- [x] `CHANGELOG.md` — dated, factual, unchanged
- [x] `DEV_NOTES.md` — internal dev journal (formerly `CLAUDE.md`)

## Dashboard screenshots (`submission/screenshots/`)

- [x] All 9 screenshots current against the live UI, verified by re-opening
      the running dashboard, not assumed from an earlier pass
- [x] Accept/reject round-trip screenshots show a genuine before/after
      (checked against `/api/trust`, not just the image)

## Submission package (`submission/`)

- [x] `submission/code/` — clean copy, excludes `.venv/`, `node_modules/`,
      `data/`, `models/checkpoints/`, `HANDOFF.md`
- [x] `submission/docs/` — architecture, evaluation report, benchmark
      report, project log
- [x] `submission/metrics/` — `metrics.json`, `benchmark.json`,
      `threshold_sweep.csv`
- [x] `submission/api-samples/` — 7 sample requests/responses, including
      one malformed payload returning a clean 4xx
- [x] `submission/pdf_backup/` — PDF fallback if the ZIP is rejected
- [ ] `submission/presentation/` — placeholder only; **the filled Honeywell
      template still needs to be added and the ZIP rebuilt after**

## Before you upload

1. Add the filled presentation to `submission/presentation/`.
2. Re-zip: `zip -r ~/Desktop/Grade_Change_Intelligence_Submission_Kamil.zip submission/ -x "*.DS_Store" -x "*/__pycache__/*"`.
3. Extract the ZIP somewhere clean and confirm it matches `submission/`
   exactly before uploading — don't trust the zip step blindly.
4. Check the HirePro portal directly for any AI-disclosure requirement —
   nothing in this repository answers that question one way or the other.
