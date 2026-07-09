# Release Checklist

Covers a full QuantScenarioBench release: the Python library (GitHub tag → PyPI),
the Zenodo DOI, and the Hugging Face assets (benchmark datasets, Evaluation
Results repo, Leaderboard Space). The library follows SemVer; `dataset_version`
is versioned independently (see `CHANGELOG.md` header and `export/_publish.py`).

Release history convention: each release is prepared on a `docs/vX.Y.Z-release-notes`
branch and merged via PR, then tagged (`v1.0.0`–`v1.2.1` all followed this).

## 1. Pre-flight

- [ ] All feature branches for this release are merged to `main`; working tree clean.
- [ ] CI is green on `main` (`.github/workflows/ci.yml`).
- [ ] Fresh-environment test run passes: `pip install -e ".[dev]" && pytest`.
- [ ] Model and strategy conformance suites pass (`tests/test_conformance.py`,
      `tests/test_benchmark_conformance.py`).
- [ ] Golden fixtures (`tests/fixtures/golden_benchmark_results.json`) unchanged —
      or every intentional metric/behavior change is called out in the CHANGELOG.
- [ ] Decide the version bump (SemVer: breaking → major, feature → minor, fix → patch).
- [ ] Triage open bugs: anything release-blocking (e.g. serialization bugs that
      corrupt published EvaluationResults) is fixed or explicitly deferred in the notes.

## 2. Release-prep PR (branch: `docs/vX.Y.Z-release-notes`)

- [ ] Bump `version` in `pyproject.toml`.
- [ ] Add the `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md`
      (Keep a Changelog format: Added / Changed / Fixed / Notes).
- [ ] Update the README: Roadmap table statuses, docs for new capabilities,
      and any version numbers quoted in examples.
- [ ] Bump the pinned library ref in `spaces/leaderboard/requirements.txt`
      (`quantscenariobench @ git+...@vX.Y.Z` — or the PyPI pin once published there).
- [ ] Verify `spaces/leaderboard/README.md` front-matter (`sdk_version`) still
      matches what the Space actually runs.
- [ ] Update `CITATION.cff` (version + release date; DOI after step 3 if versioned DOIs are cited).
- [ ] Open PR, review, merge to `main`.

## 3. Tag, GitHub Release, DOI

- [ ] `git tag vX.Y.Z && git push origin vX.Y.Z`.
- [ ] Create the GitHub Release with notes (copy from the CHANGELOG section).
- [ ] Verify Zenodo archived the release and minted the version DOI; confirm the
      concept-DOI badge in the README still resolves.

## 4. Package publish (PyPI)

> ⚠️ As of v1.2.1 the package is **not** on PyPI even though the README says
> `pip install quantscenariobench`. Until first PyPI publication, either publish
> or fix the README install instructions — don't release with a broken install path.

- [ ] Build from the tag: `python -m build` (hatchling backend); `twine check dist/*`.
- [ ] Upload (`twine upload` — or GitHub Actions trusted publishing once configured).
- [ ] Verify in a clean venv: `pip install quantscenariobench==X.Y.Z`, then import
      and run a minimal `simulate()` + `run_benchmark()` smoke test.

## 5. Hugging Face assets

- [ ] If generation code, model parameters, or the Parquet schema changed:
      regenerate benchmark sample datasets, bump `dataset_version`, publish with
      `publish_to_hub()`, and confirm the auto-generated dataset cards are correct.
- [ ] Record the Hub dataset **revision hashes** for the canonical datasets in the
      release notes so downstream benchmark users can pin them.
- [ ] Confirm the Evaluation Results repo (`QSB_EVAL_RESULTS_REPO`) is reachable
      and its documented default matches reality.
- [ ] Redeploy the Leaderboard Space: push `spaces/leaderboard/` contents to the
      Space's git repo; watch the build; confirm the table renders.
- [ ] End-to-end smoke test: run a baseline strategy on a published dataset with
      the released version, `publish_evaluation_results()`, and confirm the run
      appears on the live Space.

## 6. Post-release

- [ ] Delete the merged `docs/vX.Y.Z-release-notes` branch (local + remote).
- [ ] Close the milestone / shipped issues.
- [ ] Announce (HF org page, relevant communities).
- [ ] File follow-up issues for anything deferred during the release.
