#!/usr/bin/env bash
# Builds the hackathon submission zip under the 10 MB limit (DEV_NOTES.md).
#
# The two trained model artefacts (models/*.joblib, ~16 MB combined) are
# committed to git so the Render-hosted API has real models without a
# training step in the build -- but they do NOT belong in the graded
# submission zip, the same way data/ is regenerated rather than shipped
# (see PROJECT_LOG.md's "known packaging item"). Both are reproducible with
# `python scripts/train_risk_model.py` / `train_forecast_model.py`, so
# excluding them loses nothing a judge running the code can't regenerate.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="gci-submission.zip"
rm -f "$OUT"

git ls-files \
  | grep -v '\.joblib$' \
  | grep -v '^HANDOFF\.md$' \
  | zip -q "$OUT" -@

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
unzip -l "$OUT" | tail -1
