"""Invariants of the GitHub Actions workflows.

Workflow bugs are expensive to find the normal way: the feedback loop is a push
and a wait, and the failure often is not an error but a silently wrong outcome
-- a deploy that publishes yesterday's data, or one that never fires at all.
These check the properties that matter by reading the YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def triggers(workflow: dict) -> dict:
    """The `on:` block. YAML 1.1 parses a bare `on` as the boolean True."""
    return workflow.get("on", workflow.get(True))


def test_there_is_exactly_one_place_that_deploys():
    """Two copies of the deploy steps would drift apart, and they had begun to.

    Parsed rather than grepped: a comment mentioning actions/deploy-pages is not
    a second deployment, and treating it as one made this fail on its own
    docstring the first time round.
    """
    deploying = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in (workflow.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                if str(step.get("uses", "")).startswith("actions/deploy-pages"):
                    deploying.append(path.name)
    assert deploying == ["deploy.yml"], (
        f"deploy-pages is used in {deploying}; it belongs only in deploy.yml"
    )


def test_ci_deploys_only_after_the_tests_pass():
    """A push that breaks the suite must never reach the live page."""
    ci = load("ci.yml")
    deploy = ci["jobs"]["deploy"]
    assert deploy.get("needs") == "test"
    assert deploy["uses"].endswith("deploy.yml")


def test_ci_does_not_deploy_pull_requests_or_other_branches():
    condition = load("ci.yml")["jobs"]["deploy"]["if"]
    assert "refs/heads/main" in condition
    assert "pull_request" not in condition


def test_a_manual_ci_run_can_still_publish():
    """The escape hatch for a slow push trigger.

    GitHub queued a push-triggered run for 25 minutes on 2026-08-26. Dispatching
    CI by hand was the obvious response, and with the job gated on `push` alone
    it ran the tests and skipped the deploy -- the opposite of what was wanted.
    """
    condition = load("ci.yml")["jobs"]["deploy"]["if"]
    assert "workflow_dispatch" in condition
    assert "push" in condition


def test_ci_still_runs_on_pull_requests():
    """The gate must not have been achieved by switching CI off for PRs."""
    assert "pull_request" in triggers(load("ci.yml"))


def test_the_daily_run_deploys_the_commit_it_just_made():
    """It commits regenerated data, then deploys.

    Without an explicit ref the reusable workflow would check out the SHA that
    triggered the run -- the state *before* today's data was committed -- and
    publish fresh code against yesterday's numbers.
    """
    deploy = load("daily-forecast.yml")["jobs"]["deploy"]
    assert deploy.get("needs") == "forecast"
    assert "ref" in deploy.get("with", {}), (
        "the daily deploy must pass a ref, or it publishes pre-commit state"
    )
    assert "github.ref_name" in deploy["with"]["ref"]


def test_deploy_checks_out_a_branch_head_not_a_sha():
    step = next(
        s for s in load("deploy.yml")["jobs"]["deploy"]["steps"]
        if str(s.get("uses", "")).startswith("actions/checkout")
    )
    ref = step["with"]["ref"]
    assert "inputs.ref" in ref and "github.ref_name" in ref


def test_deploy_stamps_assets_before_uploading():
    """Order matters: stamping after upload would publish unstamped URLs."""
    steps = load("deploy.yml")["jobs"]["deploy"]["steps"]
    names = [str(s.get("name", "")) + str(s.get("uses", "")) for s in steps]
    stamp = next(i for i, n in enumerate(names) if "stamp-assets" in str(steps[i]))
    upload = next(i for i, n in enumerate(names) if "upload-pages-artifact" in n)
    assert stamp < upload


def test_deploy_verifies_the_artifact_before_publishing():
    steps = load("deploy.yml")["jobs"]["deploy"]["steps"]
    verify = next(
        (i for i, s in enumerate(steps) if "verify_site.py" in str(s.get("run", ""))),
        None,
    )
    assert verify is not None, "deploy must verify the artifact; publishing has no undo"
    upload = next(
        i for i, s in enumerate(steps)
        if "upload-pages-artifact" in str(s.get("uses", ""))
    )
    assert verify < upload, "verification must run before the artifact is uploaded"


def test_deployments_cannot_race_each_other():
    """Pages allows one deployment at a time and the two callers can overlap."""
    deploy = load("deploy.yml")
    concurrency = deploy["concurrency"]
    assert concurrency["group"] == "pages-deploy"
    assert concurrency["cancel-in-progress"] is False, (
        "cancelling a half-finished deploy leaves the live site on whichever "
        "version won the race"
    )


def test_deploy_has_the_permissions_pages_needs():
    for name, job in (("ci.yml", "deploy"), ("daily-forecast.yml", "deploy")):
        permissions = load(name)["jobs"][job]["permissions"]
        assert permissions.get("pages") == "write", name
        assert permissions.get("id-token") == "write", name


def test_the_verifier_script_exists_and_is_referenced_correctly():
    referenced = Path(__file__).resolve().parents[1] / ".github/scripts/verify_site.py"
    assert referenced.is_file()
    assert "verify_site.py" in (WORKFLOWS / "deploy.yml").read_text(encoding="utf-8")
