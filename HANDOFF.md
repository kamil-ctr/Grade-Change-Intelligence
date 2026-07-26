# Handoff to Claude Code

Everything needed to continue development from a terminal.

---

## 1. Move the project somewhere sensible

The project currently lives inside a Cowork session folder. Copy the **source
only** — `data/` is 186 MB and fully regenerable.

```bash
mkdir -p ~/projects
rsync -av --exclude 'data' --exclude '__pycache__' --exclude '.DS_Store' \
  ~/Library/Application\ Support/Claude/local-agent-mode-sessions/*/*/*/outputs/gci/ \
  ~/projects/gci/
cd ~/projects/gci
```

If the glob doesn't resolve, open the folder from the Cowork file card and copy
it manually — the destination is what matters.

Confirm you got everything:

```bash
ls          # expect: gci/ scripts/ tests/ models/ DEV_NOTES.md PROJECT_LOG.md CHANGELOG.md README.md requirements.txt
```

---

## 2. Start a git repo

Worth doing before Claude Code starts editing — it makes every change reviewable
and reversible.

```bash
git init
git add -A
git commit -m "Phase 0 + ML pipeline: twin, controller, faults, events, features, risk model"
```

`.gitignore` already excludes `data/`, checkpoints, and `.joblib` artefacts.

---

## 3. Install Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude --version
```

Requires macOS 13+ and a Pro, Max, Team, or Enterprise plan. Alternatives:
`brew install --cask claude-code`, or `npm install -g @anthropic-ai/claude-code`.

---

## 4. Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Two install gotchas that cost real time in the sandbox:

```bash
pip install xgboost-cpu      # NOT xgboost — default wheel bundles 131 MB of CUDA
pip install shap             # needs numba/llvmlite; optional, see DEV_NOTES.md
```

Verify before doing anything else:

```bash
python -m unittest discover -s tests -t .   # expect 115 tests, OK
python scripts/generate_data.py --events 500
python scripts/train_risk_model.py
```

---

## 5. Launch Claude Code

```bash
cd ~/projects/gci
claude
```

`DEV_NOTES.md` loads automatically — it contains the architecture, the
non-negotiable rules, current state, the next task, and the known traps.

---

## 6. Kickoff prompt

Paste this as your first message:

> Read DEV_NOTES.md, PROJECT_LOG.md and CHANGELOG.md to load the project state.
>
> This is a Honeywell hackathon submission due 2026-07-26 23:59. Phase 0 and
> Phase 1 module 1 (the ML pipeline and risk model) are complete with 115 tests
> passing. Do not redesign the architecture.
>
> First: I'm now on real hardware, not the constrained sandbox the earlier work
> was done in. Restore the settings listed in the "Environment note" section of
> DEV_NOTES.md — Random Forest capacity, SHAP and permutation-importance sample
> sizes — regenerate the corpus at 1500 events in one run, and retrain. Confirm
> the metrics improve or stay stable, and report event detection rate and median
> warning time.
>
> Then continue Phase 1 in this order, completing and verifying each module
> before starting the next: forecast.py → roi.py → optimizer.py → discovery.py →
> stabilization.py → provenance.py → ledger.py → api/ → frontend/.
>
> For every module: write unit tests, integrate with existing modules, keep the
> suite green, expose API endpoints where relevant, and update PROJECT_LOG.md and
> CHANGELOG.md. Do not stop between modules unless you hit an architectural
> blocker or need a decision from me.

---

## 7. Useful things once you're in

| Want | Do |
|---|---|
| Let it plan before coding | Press **Shift+Tab** twice for plan mode |
| Stop it mid-task | **Esc** |
| Go back to an earlier point | **Esc Esc**, or `/rewind` |
| Fewer permission prompts for tests | `/permissions` → allow `Bash(python -m unittest:*)` |
| Save a new convention to memory | `#` followed by the rule — it offers to write it to DEV_NOTES.md |
| Check context usage | `/context` |
| Compact a long session | `/compact` |
| Review its work | `git diff`, or the `/code-review` command |

**Long training runs:** the sandbox's 45-second shell limit is gone, but Claude
Code has its own command timeout. For anything over ~2 minutes, have it run in
the background (`&` plus a log file) and poll, or raise
`BASH_MAX_TIMEOUT_MS` in `.claude/settings.json`.

---

## 8. Time check

The remaining work is 8 modules including a React dashboard. Against the
deadline, the order in the kickoff prompt is already priority-sorted: after
`ledger.py` you have all six graded deliverables covered by working code, and
`api/` + `frontend/` make them visible.

If time gets tight, tell Claude Code to skip straight from `roi.py` to
`stabilization.py`, `provenance.py`, `ledger.py`, then the API and dashboard —
`optimizer.py` and `discovery.py` have partial coverage from existing features
(`min_feasible_ramp_min`, `bw_dev_projected`) and can be thinner.

**Reserve the last two hours** for the architecture doc, the six-slide deck,
dashboard screenshots, packaging under 10 MB, and one clean-checkout rehearsal.
That block is not optional — a great system with a deck written in the final ten
minutes scores worse than a good system presented properly.
