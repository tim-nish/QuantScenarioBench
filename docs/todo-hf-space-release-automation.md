# TODO / Issue draft: Automate Hugging Face Leaderboard Space updates on release

**Status:** Planned — deferred (blocked, see below). Draft for a future GitHub issue.

## Goal

When a QSB release is published, automatically update the hosted Leaderboard
Space to consume the newly released **PyPI** package and rebuild it — removing
the manual `spaces/leaderboard/requirements.txt` edit and manual push that is
currently step 5 of `docs/RELEASE_CHECKLIST.md` and easy to forget.

On release the automation should:

1. Rewrite the Space's `requirements.txt` to pin `quantscenariobench==X.Y.Z`
   (PyPI), matching the just-released version.
2. Commit/push that to the Space's Hugging Face git repo, which triggers an
   automatic rebuild.
3. Verify (or, at minimum, document how to verify) that the Space finishes
   building and reaches the `RUNNING` stage.

## Why this is deferred (blockers — resolve before implementing)

1. **The target Space repo ID is not finalized.** `spaces/leaderboard/README.md`
   and the PRD (Open Questions 18, 22) note the Hugging Face namespace is still
   undecided. The workflow needs a concrete `spaces/<owner>/<name>` to push to.
   **Decide this first.**
2. **Requires a Hugging Face write token as a GitHub secret** — only the
   maintainer can create it (see Required secrets).
3. **Publish ordering.** The Space rebuild runs `pip install
   quantscenariobench==X.Y.Z`, so it must run only after the PyPI upload is
   live and installable. PyPI publish is currently a manual `twine upload`
   (checklist step 4), so the workflow must either wait for PyPI or be chained
   after an automated publish job.

## Required secrets

- **`HF_TOKEN`** — a Hugging Face **write** access token, ideally a
  fine-grained token scoped to only the Leaderboard Space repo. Stored as a
  GitHub Actions repository secret. Used to authenticate the commit/push (or
  the `huggingface_hub` upload) to the Space. Must never be echoed in logs.

## Workflow trigger

Recommended: `on: release: types: [published]` — fires when the GitHub Release
is published (which the checklist does right after tagging). Preferred over
`push: tags` because a published Release is an explicit human action and won't
fire on arbitrary tags, and it naturally lands after the tag exists.

Because PyPI publish is currently manual, the workflow **must first confirm the
target version is installable from PyPI** before touching the Space — e.g. poll
`https://pypi.org/pypi/quantscenariobench/json` for the release version with a
bounded retry, and fail fast if it never appears. If/when PyPI publishing is
automated via a trusted-publishing `release.yml`, chain this Space job after it
with `needs:`/`workflow_run` instead of polling.

Also provide a `workflow_dispatch` entry point so the automation can be tested
against a scratch Space before it is trusted on real releases.

## Target files

- **New:** `.github/workflows/deploy-space.yml` (the automation).
- **Edited at runtime in the _Space's_ repo:** `requirements.txt` (the PyPI pin).
- **In this repo:** `spaces/leaderboard/requirements.txt` stays the source of
  truth for the Space's contents and is already on the PyPI pin
  (`quantscenariobench==X.Y.Z`). The workflow derives the version from the
  release tag and pushes the updated file to the Space repo.

## Implementation sketch (for when unblocked)

Two viable mechanisms:

1. **`huggingface_hub` (recommended):** `HfApi().upload_file(path_in_repo=
   "requirements.txt", repo_id="<owner>/<space>", repo_type="space",
   token=HF_TOKEN, commit_message="Pin quantscenariobench==X.Y.Z")`. The commit
   triggers an automatic rebuild.
2. **Git push:** clone `https://user:${HF_TOKEN}@huggingface.co/spaces/<owner>/<space>`,
   rewrite `requirements.txt`, commit, push.

Verify by polling `HfApi().get_space_runtime(repo_id).stage` (or `GET
https://huggingface.co/api/spaces/<repo>/runtime`) until it reaches `RUNNING`,
failing on `BUILD_ERROR`/`RUNTIME_ERROR` or a timeout. Optionally also HTTP-GET
the Space URL and assert a 200.

## Acceptance criteria

- Publishing a GitHub Release `vX.Y.Z` runs the workflow automatically.
- It refuses to proceed (fails visibly) until `quantscenariobench==X.Y.Z` is
  installable from PyPI.
- The Space's `requirements.txt` ends up pinned to `quantscenariobench==X.Y.Z`
  (PyPI, not the GitHub tag).
- The Space rebuilds and reaches `RUNNING`; the workflow reports the final stage.
- On build failure or timeout, the workflow fails with a log pointing at the
  Space's build logs.
- No secret is printed in logs.
- A `workflow_dispatch` dry-run path exists for testing against a scratch Space.

## Rollback plan

- The workflow only edits the Space's `requirements.txt`. To roll back, revert
  that one commit in the Space repo (the Space keeps its own git history) and it
  rebuilds against the previous pin.
- If the automation misbehaves, disable it by removing/renaming
  `.github/workflows/deploy-space.yml` or disabling the workflow in the Actions
  tab; the manual checklist step remains the unchanged fallback.
- The Space and the library are separate artifacts, so a bad Space deploy never
  affects the published PyPI package, the GitHub Release, or the Zenodo archive.

## Until this exists

Follow `docs/RELEASE_CHECKLIST.md` step 5 manually: after PyPI is live, ensure
`spaces/leaderboard/requirements.txt` pins `quantscenariobench==X.Y.Z` and push
`spaces/leaderboard/` to the Space repo, then confirm the Space rebuilds and the
table renders.
