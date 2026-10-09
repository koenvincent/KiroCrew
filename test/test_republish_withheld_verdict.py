"""Unit tests for .github/scripts/republish_withheld_verdict.py.

The script re-runs the advisory review lane(s) whose withheld verdict never
reached a PR. These tests drive it with ``gh`` and the disposition gate replaced
by stubs, so the lane-location logic and the fail-closed behaviour are verified
without a live forge.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "republish_withheld_verdict.py"
OVERRIDE_HANDLER = REPO_ROOT / ".github" / "workflows" / "ai-review-human-override.yml"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="requires the republisher script")


def _load():
    spec = importlib.util.spec_from_file_location("republish_withheld_verdict", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load()

REPO = "kirodotdev/KiroCrew"
HEAD = "4328fd0f941f09ff10f245fbdb4accf7c246febe"


class _Recorder:
    """Replaces the module's gh / gate / rerun calls with scripted answers and
    records every rerun the script requested."""

    def __init__(
        self,
        monkeypatch,
        *,
        owed,
        same_repo_runs=None,
        fork_rows=None,
        fork_run_path=None,
        fork_run_attempt=1,
    ):
        self.rerun_calls: list[str] = []
        self._owed = owed
        self._same_repo_runs = same_repo_runs or []
        self._fork_rows = fork_rows or []
        self._fork_run_path = fork_run_path
        self._fork_run_attempt = fork_run_attempt

        monkeypatch.setattr(MOD, "owed_lanes", lambda repo, pr, head: self._owed)
        monkeypatch.setattr(MOD, "_gh_json", self._gh_json)
        monkeypatch.setattr(MOD, "_rerun", self._rerun)
        # Default: a same-repo PR with a readable head. Tests override as needed.
        monkeypatch.setattr(
            MOD, "_resolve_pr_meta", lambda repo, pr: (False, repo, "feature-branch")
        )

    def _rerun(self, repo, run_id):
        self.rerun_calls.append(run_id)
        return True

    def _gh_json(self, args):
        joined = " ".join(args)
        if "/actions/runs?" in joined:
            return {"workflow_runs": self._same_repo_runs}
        if "/check-runs?" in joined:
            return {"check_runs": self._fork_rows}
        if "/actions/runs/" in joined and self._fork_run_path is not None:
            return {"path": self._fork_run_path, "run_attempt": self._fork_run_attempt}
        return None


def test_no_owed_lane_reruns_nothing(monkeypatch, capsys):
    rec = _Recorder(monkeypatch, owed=[])
    assert MOD.republish(REPO, "7", HEAD, is_fork=False, head_repo=REPO, head_ref="b") == 0
    assert rec.rerun_calls == []
    assert "no advisory lane owes" in capsys.readouterr().out


def test_unreadable_gate_reruns_nothing(monkeypatch, capsys):
    rec = _Recorder(monkeypatch, owed=None)
    assert MOD.republish(REPO, "7", HEAD, is_fork=False, head_repo=REPO, head_ref="b") == 0
    assert rec.rerun_calls == []
    assert "could not read this PR's comments" in capsys.readouterr().out


def test_same_repo_owed_lane_is_rerun_by_newest_matching_run(monkeypatch):
    runs = [
        {
            "id": 100,
            "path": ".github/workflows/design-review.yml",
            "head_repository": {"full_name": REPO},
            "head_branch": "feature-branch",
        },
        {
            "id": 200,  # newer, same lane/binding -> the one re-run
            "path": ".github/workflows/design-review.yml",
            "head_repository": {"full_name": REPO},
            "head_branch": "feature-branch",
        },
        {
            "id": 300,  # a different lane, must be ignored
            "path": ".github/workflows/ux-review.yml",
            "head_repository": {"full_name": REPO},
            "head_branch": "feature-branch",
        },
    ]
    rec = _Recorder(monkeypatch, owed=["DESIGN"], same_repo_runs=runs)
    assert (
        MOD.republish(REPO, "7", HEAD, is_fork=False, head_repo=REPO, head_ref="feature-branch")
        == 1
    )
    assert rec.rerun_calls == ["200"]


def test_same_repo_binding_rejects_a_sibling_prs_run(monkeypatch, capsys):
    # Same head SHA, but a DIFFERENT head repo/branch -> not this PR's run.
    runs = [
        {
            "id": 200,
            "path": ".github/workflows/design-review.yml",
            "head_repository": {"full_name": "someone/fork"},
            "head_branch": "other",
        }
    ]
    rec = _Recorder(monkeypatch, owed=["DESIGN"], same_repo_runs=runs)
    assert (
        MOD.republish(REPO, "7", HEAD, is_fork=False, head_repo=REPO, head_ref="feature-branch")
        == 0
    )
    assert rec.rerun_calls == []
    assert "could not be located" in capsys.readouterr().out


def test_fork_lane_is_rerun_from_its_lane_run_marker(monkeypatch):
    rows = [
        {
            "id": 10,
            "output": {"text": "Lane run: ...\n\n<!-- ai-review-fork-lane run=55501 -->"},
        }
    ]
    rec = _Recorder(
        monkeypatch,
        owed=["UX"],
        fork_rows=rows,
        fork_run_path=".github/workflows/fork-ux-review.yml",
    )
    assert MOD.republish(REPO, "7", HEAD, is_fork=True, head_repo="someone/fork", head_ref="b") == 1
    assert rec.rerun_calls == ["55501"]


def test_fork_lane_marker_pointing_at_the_wrong_workflow_is_refused(monkeypatch, capsys):
    rows = [
        {
            "id": 10,
            "output": {"text": "<!-- ai-review-fork-lane run=55501 -->"},
        }
    ]
    rec = _Recorder(
        monkeypatch,
        owed=["UX"],
        fork_rows=rows,
        fork_run_path=".github/workflows/some-other.yml",  # not the fork UX lane
    )
    assert MOD.republish(REPO, "7", HEAD, is_fork=True, head_repo="someone/fork", head_ref="b") == 0
    assert rec.rerun_calls == []
    assert "could not be located" in capsys.readouterr().out


def test_same_repo_run_at_the_attempt_cap_is_not_rerun(monkeypatch, capsys):
    """The blocking finding: a lane whose withhold cause persists must stop being
    re-run once it has reached MAX_RERUN_ATTEMPTS, so the sweep cannot re-fire a
    paid model review every STALE_MINUTES forever. Fails if the cap is removed."""
    runs = [
        {
            "id": 200,
            "run_attempt": MOD.MAX_RERUN_ATTEMPTS,  # already at the cap
            "path": ".github/workflows/design-review.yml",
            "head_repository": {"full_name": REPO},
            "head_branch": "feature-branch",
        }
    ]
    rec = _Recorder(monkeypatch, owed=["DESIGN"], same_repo_runs=runs)
    assert (
        MOD.republish(REPO, "7", HEAD, is_fork=False, head_repo=REPO, head_ref="feature-branch")
        == 0
    )
    assert rec.rerun_calls == []
    out = capsys.readouterr().out
    assert "attempts without publishing" in out
    assert "manually" in out


def test_same_repo_run_below_the_cap_is_still_rerun(monkeypatch):
    """One attempt below the cap must still re-run -- the bound stops a stuck
    lane, it does not stop the recovery itself."""
    runs = [
        {
            "id": 200,
            "run_attempt": MOD.MAX_RERUN_ATTEMPTS - 1,
            "path": ".github/workflows/design-review.yml",
            "head_repository": {"full_name": REPO},
            "head_branch": "feature-branch",
        }
    ]
    rec = _Recorder(monkeypatch, owed=["DESIGN"], same_repo_runs=runs)
    assert (
        MOD.republish(REPO, "7", HEAD, is_fork=False, head_repo=REPO, head_ref="feature-branch")
        == 1
    )
    assert rec.rerun_calls == ["200"]


def test_fork_run_at_the_attempt_cap_is_not_rerun(monkeypatch, capsys):
    """The attempt bound holds on the fork path too, read from the same
    actions/runs/{id} payload the fork-workflow verification already fetches."""
    rows = [
        {
            "id": 10,
            "output": {"text": "<!-- ai-review-fork-lane run=55501 -->"},
        }
    ]
    rec = _Recorder(
        monkeypatch,
        owed=["UX"],
        fork_rows=rows,
        fork_run_path=".github/workflows/fork-ux-review.yml",
        fork_run_attempt=MOD.MAX_RERUN_ATTEMPTS,
    )
    assert MOD.republish(REPO, "7", HEAD, is_fork=True, head_repo="someone/fork", head_ref="b") == 0
    assert rec.rerun_calls == []
    assert "attempts without publishing" in capsys.readouterr().out


def test_same_repo_binding_rejects_a_run_in_a_sibling_head_repo(monkeypatch, capsys):
    """The head_repository.full_name binding must reject a run that matches the
    workflow, head SHA and branch but belongs to a DIFFERENT head repository
    (the shared-fork-head case). Fails when that binding is removed."""
    runs = [
        {
            "id": 200,
            "run_attempt": 1,
            "path": ".github/workflows/design-review.yml",
            # Same branch name and lane, but a different head repo -> not ours.
            "head_repository": {"full_name": "someone/fork"},
            "head_branch": "feature-branch",
        }
    ]
    rec = _Recorder(monkeypatch, owed=["DESIGN"], same_repo_runs=runs)
    assert (
        MOD.republish(REPO, "7", HEAD, is_fork=False, head_repo=REPO, head_ref="feature-branch")
        == 0
    )
    assert rec.rerun_calls == []
    assert "could not be located" in capsys.readouterr().out


def test_owed_lanes_drops_names_this_script_cannot_rerun(monkeypatch):
    """Only lanes in LANES are returned; a required lane the predicate might name
    is not silently treated as handled here."""
    import subprocess

    class _Proc:
        returncode = 0
        stdout = json.dumps({"ok": True, "unpublished": ["DESIGN", "OPUS"]})

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    assert MOD.owed_lanes(REPO, "7", HEAD) == ["DESIGN"]


def test_owed_lanes_fails_closed_when_the_gate_is_not_ok(monkeypatch):
    import subprocess

    class _Proc:
        returncode = 0
        stdout = json.dumps({"ok": False, "unpublished": []})

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    assert MOD.owed_lanes(REPO, "7", HEAD) is None


def test_the_lane_table_matches_the_override_handler() -> None:
    """The lane -> workflow mapping is duplicated from ai-review-human-override.yml;
    pin the two so a lane rename cannot drift this script's re-run target."""
    if not OVERRIDE_HANDLER.exists():
        pytest.skip("requires the override handler workflow")
    handler = OVERRIDE_HANDLER.read_text(encoding="utf-8")
    expected = {
        "DESIGN": ("Design Review", "design-review.yml", "fork-design-review.yml"),
        "UX": ("UX Review", "ux-review.yml", "fork-ux-review.yml"),
        "FIRST-PRINCIPLES": (
            "First Principles Review",
            "first-principles-review.yml",
            "fork-first-principles-review.yml",
        ),
    }
    for name, (check_name, same_repo, fork) in expected.items():
        spec = MOD.LANES[name]
        assert spec["check_name"] == check_name
        assert spec["same_repo"] == same_repo
        assert spec["fork"] == fork
        # The handler re-runs the same check name and fork workflow for this lane.
        assert check_name in handler
        assert same_repo in handler
        assert fork in handler
    # And the three advisory lanes are the whole set this script re-runs: the
    # required lanes (Opus/GPT/Security Scope) fail CLOSED on a withheld verdict
    # and are recovered by their own finalize, never by this path.
    assert set(MOD.LANES) == set(expected)
