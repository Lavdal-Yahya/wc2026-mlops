# Task Distribution — WC 2026 Forecasting MLOps Project

Split: **Lavdal 55% · Souleyman 15% · Mohamed 15% · Mounaa 15%**

Legend: 🔗 = dependency on another member's task. Everything not marked 🔗 can start **immediately** after the 30-min kickoff.

Parallelism rule: **all scripts read `MLFLOW_TRACKING_URI` from the environment** and run against a local `mlflow ui` until the EC2 server is up. Nobody waits on infra. The existing `src/` modules are reused; teammates mostly write thin MLflow-instrumented `scripts/` wrappers around them.

---

## Lavdal — 55% (infra, pipeline wiring, deployment, integration)

Branch: `feat/infra-flask-lavdal`

| # | Task | Depends on |
|---|---|---|
| L1 | Create GitHub repo, scaffold layout, **port existing `src/` + `configs/config.yaml` → `params.yaml`**, `requirements.txt`, `.env.example`, `.gitignore`; branch/PR workflow; add members + prof | — |
| L2 | AWS: S3 bucket (`raw/`, `processed/`, `models/`), IAM user + keys for the team, security groups | — |
| L3 | EC2 #1: install + run MLflow Tracking server (S3 artifact store, sqlite backend), open port 5000, share URI | L2 |
| L4 | DVC: `dvc init`, S3 remote, write `dvc.yaml` (preprocess → train → evaluate) wired to the `scripts/` entrypoints via contract paths/params | L2, 🔗 S1/M1/Mo1 (contracts fixed at kickoff → wiring written in parallel, tested once scripts land) |
| L5 | Version data **v1 and v2** with DVC; verify switching (`git checkout` + `dvc checkout`) | L4, 🔗 S2 (Souleyman delivers the two feature variants) |
| L6 | Flask app: **home/away dropdowns + neutral toggle** → loads latest `wc-outcome-model` from Model Registry **+ `ratings_snapshot.joblib`** → builds feature vector server-side → shows P(win/draw/loss) | 🔗 M2 (registered model) **and** 🔗 S1 (ratings snapshot). Develop against a dummy model + dummy snapshot until both land |
| L7 | EC2 #2: deploy Flask (gunicorn + systemd/nohup), open port, verify live prediction | L2, L6 |
| L8 | Integration: review + merge PRs (`dev → main`), run full pipeline on EC2 MLflow, fix cross-module issues | 🔗 all member PRs |
| L9 | Architecture schema (README + report), demo orchestration, final link verification, add prof | L8 |

## Souleyman — 15% (data & feature pipeline)

Branch: `feat/preprocess-souleyman` — min. 3 commits

| # | Task | Depends on |
|---|---|---|
| S1 | `scripts/preprocess.py` wrapping `features/build.py`: pull raw match data, build **v1 (Elo-only)** features, time-aware train/test split (cut 2023-01-01), write `data/processed/*.csv` **and `ratings_snapshot.joblib`** (latest Elo/rolling per team, for Flask); log dataset params/stats to MLflow | — (local MLflow) |
| S2 | Produce **data v2**: add rolling form + head-to-head features (behind a `params.yaml` flag) → notify Lavdal for DVC versioning (🔗 feeds L5) | S1 |
| S3 | Write "problem + dataset" section (≤1 page, incl. why log loss over accuracy) for report + README data section | S1 |

## Mohamed — 15% (training & model registry)

Branch: `feat/train-mohamed` — min. 3 commits

| # | Task | Depends on |
|---|---|---|
| M1 | `scripts/train.py` wrapping `models/baseline.py` + `models/trees.py`: train **LR / RF / GB** with expanding-window CV; log params + log loss + Brier + accuracy + macro F1 + model artifacts to MLflow | 🔗 S1 output schema (contract-fixed → develop on a stub CSV with agreed columns until real data lands) |
| M2 | Model Registry: compare runs by **mean log loss**, **register only the best** as `wc-outcome-model` (🔗 unblocks L6) | M1 |
| M3 | MLflow screenshots (experiments, run comparison, registry) for README + report | M2, 🔗 L3 (server live for real screenshots) |

## Mounaa — 15% (evaluation, docs, presentation)

Branch: `feat/evaluate-mounaa` — min. 3 commits

| # | Task | Depends on |
|---|---|---|
| Mo1 | `scripts/evaluate.py` using `evaluation/metrics.py`: load registered model, evaluate on test set (log loss, Brier, accuracy, macro F1), **calibration curve + confusion matrix**, log metrics + plots to MLflow. Optional stretch: run `simulation/runner.py` for a WC champion table | 🔗 contract only (model name + test path) — stub until M2 lands |
| Mo2 | Full README: problem, architecture (schema from L9), reproduction, screenshots (Flask/DVC from Lavdal/Souleyman, MLflow from Mohamed) | 🔗 L9, M3, S3 |
| Mo3 | Mini-report (1–5 pages): links + Souleyman's dataset section + Lavdal's schema; slides skeleton for the 25-min demo | 🔗 S3, L9 |

---

## Dependency Graph (critical path)

```
L1 (repo+src) ──► everyone clones & branches
L2 (AWS) ──► L3 (MLflow EC2) ──► everyone switches URI
                                   │
S1 (preprocess + snapshot) ──► S2 (data v2) ──► L5 (DVC versions)
      │                    └──────────────────────────► L6 (needs snapshot)
      └──► M1 (train) ──► M2 (register best) ──────────► L6 (Flask) ──► L7 (deploy EC2 #2)
                                │
                                └──► Mo1 (evaluate)
All PRs ──► L8 (integration) ──► L9 (schema/demo) ──► Mo2/Mo3 (docs/report)
```

**Critical path = Lavdal:** L1 → L2 → L3 → (L4/L5 ∥ L6) → L7 → L8 → L9.
Note L6 (Flask) now has **two** upstream deps — Mohamed's registered model *and* Souleyman's ratings snapshot — because the app builds features server-side from team names.
