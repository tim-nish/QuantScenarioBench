---
title: QuantScenarioBench Leaderboard
emoji: 🏆
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
suggested_hardware: cpu-basic
pinned: false
short_description: Ranked, sortable, filterable QuantScenarioBench Leaderboard
---

# QuantScenarioBench Leaderboard

A hosted, browsable Leaderboard for [QuantScenarioBench](https://github.com/tim-nish/QuantScenarioBench) (PRD Feature 4.10, Epic 8). This Space is a presentation layer only (FR-36): it renders the existing `EvaluationResult`/Leaderboard aggregation pipeline (Epic 7, FR-30–FR-34) and adds no aggregation, ranking, or data-model logic of its own — see `app.py`.

## Configuration

- **`QSB_EVAL_RESULTS_REPO`** (environment variable, optional) — the Hugging Face dataset repo ID this Space reads published `EvaluationResult`s from. Defaults to `quantscenariobench/evaluation-results`. The Hugging Face namespace/naming convention for that repo is still undecided (PRD Open Questions 18, 22) — set this variable once a real namespace is chosen; no code change is required.

## Compute tier (AD-29)

`suggested_hardware: cpu-basic` above is **documentation of intent, not an enforcement mechanism** — per Hugging Face's own [Spaces configuration reference](https://huggingface.co/docs/hub/spaces-config-reference), `suggested_hardware` "will not automatically assign a hardware to this Space." New Spaces default to the free `cpu-basic` tier unless explicitly upgraded via the Space's Settings page or the `huggingface_hub` API. This Space performs no model inference — only a Hub dataset read plus `pandas`/Gradio rendering, sorting, and filtering — so **no hardware upgrade should be requested**; doing so would be the one Epic 8 decision with a real recurring cost, and should go through the maintainer explicitly rather than happening by default (see Architecture Spine Deferred section).

## Deployment (AD-30)

Standard Hugging Face Space git-push-to-deploy, mirroring how Benchmark Datasets (Feature 4.4) and the Evaluation Results repo (Feature 4.9) already publish to the Hub: push the contents of this `spaces/leaderboard/` directory to the Space's own git repo. `requirements.txt` is self-sufficient for a clean build — no additional manual configuration is required beyond this README's `sdk`/`app_file` declarations. A CI-triggered push on release/tag is a deferred refinement, not part of this story.
