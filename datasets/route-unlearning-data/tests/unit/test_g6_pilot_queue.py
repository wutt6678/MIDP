"""CPU tests for the G6.1 pilot launch queue.

The queue exists so that a run cannot start against a condition that was true
when the queue was started and false when the GPU finally appeared.  These tests
cover the two parts that are easy to get subtly wrong:

- capacity must be STABLE, not momentary.  A single nvidia-smi sample can catch
  another job between allocations, and launching into that gap OOMs after the
  3000-step baseline route h has already been paid for;
- the frozen manifests are hashed before the wait and again immediately before
  the launch, and a change aborts.  Once pilot execution begins those files are
  evidence, not inputs to regenerate.

No GPU, no model and no subprocess launches: nvidia-smi, the clock and sleep are
all patched.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture(scope="module")
def q():
    spec = importlib.util.spec_from_file_location(
        "g6_pilot_queue_under_test", SCRIPTS / "g6_pilot_queue.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def fake_clock(q, monkeypatch):
    """A clock that advances 1s per read, with sleep neutered.

    ``wait_for_capacity`` reads the clock once for ``started`` and once per
    deadline check, so a counter is enough to make the deadline reachable
    without actually waiting.
    """
    ticks = iter(range(1_000_000))
    monkeypatch.setattr(q.time, "time", lambda: next(ticks))
    monkeypatch.setattr(q.time, "sleep", lambda _s: None)


def test_a_momentary_gap_never_counts_as_capacity(q, monkeypatch, fake_clock):
    """Alternating free/occupied: the streak resets every other poll, so the
    requirement is never met and the wait gives up rather than launching."""
    seq = iter([{0: 40000}, {0: 100}, {0: 40000}, {0: 100}, {0: 40000}])
    monkeypatch.setattr(q, "gpu_free_mb", lambda: next(seq, {0: 100}))
    assert q.wait_for_capacity(26000, poll_sec=1, stable_polls=2,
                               deadline_sec=4) is None


def test_a_stable_gpu_is_chosen_and_the_roomiest_one_wins(q, monkeypatch,
                                                          fake_clock):
    monkeypatch.setattr(q, "gpu_free_mb",
                        lambda: {0: 27000, 2: 44000, 3: 100})
    # GPU 0 and GPU 2 both clear the threshold; headroom decides
    assert q.wait_for_capacity(26000, poll_sec=1, stable_polls=2,
                               deadline_sec=100) == "cuda:2"
    # excluding the roomiest falls back to the next eligible
    assert q.wait_for_capacity(26000, poll_sec=1, stable_polls=2,
                               deadline_sec=100, exclude={2}) == "cuda:0"


def test_one_stable_poll_is_enough_when_asked_for(q, monkeypatch, fake_clock):
    monkeypatch.setattr(q, "gpu_free_mb", lambda: {1: 30000})
    assert q.wait_for_capacity(26000, poll_sec=1, stable_polls=1,
                               deadline_sec=100) == "cuda:1"


def test_a_missing_nvidia_smi_is_no_capacity_not_a_crash(q, monkeypatch,
                                                         fake_clock):
    monkeypatch.setattr(q, "gpu_free_mb", dict)
    assert q.wait_for_capacity(26000, poll_sec=1, stable_polls=1,
                               deadline_sec=2) is None


def test_the_launch_is_aborted_if_a_frozen_manifest_moved_while_waiting(
        q, monkeypatch, caplog):
    """The guard the launch sequence depends on: the two G6.0 artifacts must be
    byte-identical at launch time to what they were when the queue started."""
    before = {"mllmu_hierarchy.json": "a" * 64,
              "mllmu_target_selection.json": "b" * 64}
    after = dict(before, **{"mllmu_hierarchy.json": "c" * 64})
    calls = iter([before, after])
    monkeypatch.setattr(q, "frozen_hashes", lambda: next(calls))
    monkeypatch.setattr(q, "wait_for_capacity",
                        lambda *a, **k: "cuda:0")
    monkeypatch.setattr(q, "preflight",
                        lambda expect_commit=None: {"ok": True,
                                                    "problems": [],
                                                    "head": "deadbeef"})
    launched = []
    monkeypatch.setattr(q.subprocess, "run",
                        lambda *a, **k: launched.append(a) or 0)
    rc = q.main(["--dry-run"])
    assert rc == 1
    assert not launched, "the runner must not be invoked after a refusal"
    assert "changed while waiting" in caplog.text


def test_a_dirty_tree_or_a_failed_verification_blocks_the_launch(
        q, monkeypatch):
    monkeypatch.setattr(q, "git_dirty_tracked",
                        lambda: [" M scripts/e2c_v3_mllmu_matrix.py"])
    res = q.preflight()
    assert res["ok"] is False
    assert any("dirty" in p for p in res["problems"])

    monkeypatch.setattr(q, "git_dirty_tracked", list)
    monkeypatch.setattr(q, "git_head", lambda: "f" * 40)
    res2 = q.preflight(expect_commit="b29fadc")
    assert res2["ok"] is False
    assert any("expected b29fadc" in p for p in res2["problems"])


def test_the_verifiers_run_in_their_own_process(q, monkeypatch):
    """A subprocess, not an import: the committed verifier is the authority on
    its own artifact, and this script's module cache must not get a vote."""
    # The git helpers also go through subprocess.run, and patching it wholesale
    # would make them report a dirty tree ("{}" parses as one non-empty line),
    # so preflight would bail out before ever reaching the verifiers.
    monkeypatch.setattr(q, "git_dirty_tracked", list)
    monkeypatch.setattr(q, "git_head", lambda: "a" * 40)
    seen = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs.get("cwd"), kwargs.get("env", {})
                     .get("CUDA_VISIBLE_DEVICES")))

        class R:
            returncode = 0
            stdout = "{}"
            stderr = ""
        return R()
    monkeypatch.setattr(q.subprocess, "run", fake_run)
    assert q.preflight()["ok"] is True
    scripts = [a[0][1:] for a in seen]
    assert any("e2c_v3_mllmu_hierarchy.py" in s[0] for s in scripts)
    assert any("e2c_v3_mllmu_matrix.py" in s[0] for s in scripts)
    # CPU-only, and run from the dataset root
    assert all(cpu == "" for _, _, cpu in seen)
    assert all(str(cwd).endswith("route-unlearning-data")
               for _, cwd, _ in seen)


def test_the_launch_command_is_the_frozen_five_set_pilot(q):
    cmd = q.launch_command("cuda:2")
    assert cmd[1].endswith("e2c_v3_granularity_matrix.py")
    assert cmd[2:4] == ["--dataset", "mllmu"]
    assert cmd[4:6] == ["--phase", "all"]
    assert cmd[-2:] == ["--device", "cuda:2"]
    # the pilot runs the frozen seeds; nothing here narrows it to a subset
    assert "--smoke" not in cmd
    assert "--only-seeds" not in cmd
    assert "--only-sets" not in cmd
