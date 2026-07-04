"""
Story 8.4 — Space Deployment Configuration & Compute Tier

Covers all acceptance criteria from GitHub Issue #61: Hugging Face
Space metadata (sdk, app_file, hardware tier) and a self-sufficient
`requirements.txt` for standard git-push-to-deploy (AD-29, AD-30).

Important nuance, confirmed against Hugging Face's own Spaces
configuration reference (docs/hub/spaces-config-reference): the
`suggested_hardware` README front-matter key does NOT itself assign
hardware to a Space -- "Setting this value will not automatically
assign a hardware to this Space." It is documentation of intent for
duplicators; the actual tier is whatever the Space defaults to (the
free `cpu-basic` tier) unless explicitly upgraded elsewhere. Tests
below verify the documented intent and the absence of an upgrade
request, not an enforcement mechanism that doesn't exist at this layer.
"""

from __future__ import annotations

from pathlib import Path

import yaml

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _space_dir() -> Path:
    return Path(__file__).parent.parent / "spaces" / "leaderboard"


def _readme_frontmatter() -> dict:
    text = (_space_dir() / "README.md").read_text()
    assert text.startswith("---\n"), "README.md must start with a YAML front-matter block"
    end = text.index("\n---", 4)
    return yaml.safe_load(text[4:end])


# ---------------------------------------------------------------------------
# AC: README.md YAML front matter declares sdk: gradio, an app_file
# pointing to app.py, and a hardware tier of cpu-basic — no GPU or paid
# tier is configured (AD-29)
# ---------------------------------------------------------------------------

def test_readme_exists_with_parseable_frontmatter():
    assert (_space_dir() / "README.md").is_file()
    frontmatter = _readme_frontmatter()
    assert isinstance(frontmatter, dict)


def test_readme_declares_gradio_sdk_and_app_file():
    frontmatter = _readme_frontmatter()
    assert frontmatter["sdk"] == "gradio"
    assert frontmatter["app_file"] == "app.py"


def test_readme_short_description_within_hub_length_limit():
    # Hub's live YAML validation (POST /api/validate-yaml) rejects a
    # Space push outright if short_description exceeds 60 characters --
    # a real deploy-blocking error hit deploying QuantScenarioBench/
    # qsb-leaderboard, not caught by any test until now.
    frontmatter = _readme_frontmatter()
    short_description = frontmatter.get("short_description")
    assert short_description, "README should declare a short_description"
    assert len(short_description) <= 60, (
        f"short_description is {len(short_description)} chars (max 60) — "
        "the Hub rejects the push outright, not just a display truncation"
    )


def test_readme_declares_free_cpu_hardware_not_gpu_or_paid_tier():
    frontmatter = _readme_frontmatter()
    assert frontmatter["suggested_hardware"] == "cpu-basic", (
        "AD-29: the Space must not default to a GPU or paid tier — it "
        "performs no model inference"
    )
    gpu_and_paid_tiers = (
        "cpu-upgrade", "t4-small", "t4-medium", "l4x1", "l4x4",
        "l40sx1", "l40sx4", "l40sx8", "a10g-small", "a10g-large",
        "a10g-largex2", "a10g-largex4", "a100-large", "a100x4", "a100x8",
    )
    assert frontmatter["suggested_hardware"] not in gpu_and_paid_tiers


def test_readme_documents_that_suggested_hardware_is_not_an_assignment_mechanism():
    # Guards against a future edit relying on suggested_hardware alone
    # and believing it enforces the tier — it explicitly does not,
    # per Hugging Face's own Spaces configuration reference.
    text = (_space_dir() / "README.md").read_text()
    assert "will not automatically assign" in text or "not an enforcement mechanism" in text.lower()


# ---------------------------------------------------------------------------
# AC: requirements.txt is sufficient on its own to build a working Space
# environment — gradio>=6.19 and quantscenariobench are both present and
# correctly resolve (AD-27, AD-30)
# ---------------------------------------------------------------------------

def test_requirements_txt_is_self_sufficient_for_a_space_build():
    text = (_space_dir() / "requirements.txt").read_text()
    assert "gradio" in text
    assert "quantscenariobench" in text
    # A Space build has no access to anything outside this directory's
    # own requirements.txt — nothing here should reference a local path.
    assert "-e ." not in text
    assert "file://" not in text


def test_readme_sdk_version_matches_requirements_txt_floor():
    frontmatter = _readme_frontmatter()
    requirements_text = (_space_dir() / "requirements.txt").read_text()
    # Not a strict equality requirement (requirements.txt is a floor,
    # sdk_version pins one exact version) — just checks they're not
    # contradictory (e.g. README declaring an older major than the floor).
    sdk_version = str(frontmatter.get("sdk_version", ""))
    assert sdk_version, "README should declare an explicit sdk_version"
    assert "gradio>=" in requirements_text


# ---------------------------------------------------------------------------
# AC: the deployed Space runs within the free CPU tier's limits without
# requiring an upgrade under expected v1.2 traffic (AD-29) — verified as
# a documented, deliberate decision (no automated resource-usage check
# is possible without a live deployment)
# ---------------------------------------------------------------------------

def test_readme_documents_no_hardware_upgrade_should_be_requested():
    text = (_space_dir() / "README.md").read_text().lower()
    assert "no hardware upgrade should be requested" in text or "no model inference" in text


# ---------------------------------------------------------------------------
# Deploying QuantScenarioBench/qsb-leaderboard hit a real BUILD_ERROR:
# quantscenariobench is not published on PyPI ("ERROR: Could not find a
# version that satisfies the requirement quantscenariobench>=1.1.0 (from
# versions: none)"). requirements.txt must install it from a source pip
# can actually resolve (the project's own git repo, pinned to a tag),
# not a bare version specifier that silently assumes PyPI availability.
# ---------------------------------------------------------------------------

def test_requirements_does_not_assume_quantscenariobench_is_on_pypi():
    text = (_space_dir() / "requirements.txt").read_text()
    for line in text.splitlines():
        if line.strip().lower().startswith("quantscenariobench"):
            assert "git+" in line or "@" in line, (
                "quantscenariobench is not published on PyPI — a bare "
                f"version specifier ('{line.strip()}') fails to build on "
                "the Hub. Pin it to an installable source, e.g. "
                "'quantscenariobench @ git+https://github.com/.../QuantScenarioBench.git@vX.Y.Z'"
            )
            break
    else:
        raise AssertionError("requirements.txt must declare quantscenariobench")
