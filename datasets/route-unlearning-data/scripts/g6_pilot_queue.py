#!/usr/bin/env python3
"""Wait for GPU capacity, then launch the frozen G6.1 five-set MLLMU pilot.

This exists because the pilot is frozen, wired and tested but cannot run yet:
every GPU on the box is saturated.  Rather than have someone poll by hand and
launch at an arbitrary moment -- which is how a run ends up starting against a
dirty tree or a stale artifact -- the launch is gated on conditions that are
checkable, and the check happens immediately before the launch, not when the
queue was started.

WHAT IT REFUSES TO LAUNCH AGAINST

1. Insufficient memory, or memory that only just became sufficient.  A GPU must
   hold the required free memory for ``--stable-polls`` consecutive polls before
   it counts.  A single sample can catch another job between allocations, and a
   launch into that gap dies with an OOM partway through training -- after the
   baseline route h has already been paid for.
2. A dirty tracked worktree.  The runner gates on this too, but the queue
   reports it before consuming a GPU slot rather than after.
3. A G6.0 hierarchy or G6.1 pilot design that does not verify.  Both are
   re-verified by their own modules, in their own processes, so the queue cannot
   pass them by importing a stale copy.
4. A changed frozen manifest.  The two G6.0 artifacts are hashed before the
   preflight and again immediately before the launch, and the launch is aborted
   if they moved.  Once pilot execution begins those files are evidence, not
   inputs to be regenerated.
5. An unexpected HEAD, when ``--expect-commit`` is given.

Use ``--dry-run`` to run every check and print the exact command without
executing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("g6_pilot_queue")

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
PY = sys.executable

HIERARCHY = DATASET_ROOT / "e2c_mllmu" / "manifests" / "mllmu_hierarchy.json"
SELECTION = DATASET_ROOT / "e2c_mllmu" / "manifests" / \
    "mllmu_target_selection.json"
G6_MANIFEST = DATASET_ROOT / "e2c_mllmu" / "manifests" / \
    "mllmu_g6_manifest.json"
G6_MATRIX = DATASET_ROOT / "e2c_mllmu" / "manifests" / "matrix_mllmu.json"
FROZEN = (HIERARCHY, SELECTION, G6_MANIFEST, G6_MATRIX)

# Measured, not estimated.  A probe on GPU 1 (2026-09-10) ran this runner's own
# code path -- rv.create_adapter_model + rv.attach_lora on Qwen3.5-9B bf16,
# rv.train_supervised (the call every training phase makes: route_h, oracles and
# cells alike), and the single-sequence backend.generate(max_new_tokens=8) that
# _hard_eval issues per identity -- and recorded torch.cuda.max_memory_reserved
# of 18168 MiB, flat: 17972 after load, 18168 at the training peak, 18168 at the
# evaluation peak.  Training adds only ~196 MiB over the load because LoRA rank 8
# leaves 1.97M trainable params and make_loader uses batch_size=1 over 31-token
# sequences, and evaluation adds nothing at all.  The peak IS the model load.
#
# The 26000 MiB this replaced was a guess from "18.7-24.6 GB resident", which
# came from image-bearing MLLMU paths.  This runner is association-level only
# (image=None throughout), so that upper bound does not describe it, and waiting
# for 26 GB on a box whose GPUs peak near 20 GB free means waiting for a
# co-tenant to leave rather than for capacity to appear.
#
# The cushion is ~10%: it absorbs allocator fragmentation and a co-tenant that
# grows slightly between the stability check and the load.  It cannot absorb a
# co-tenant that grows by gigabytes mid-run -- no launch threshold can -- but an
# OOM there costs only the cell in progress, since every oracle and every
# completed cell is cached and the run resumes rather than restarting.
MEASURED_PEAK_MIB = 18168
MEASURED_CUSHION_MIB = 1832
REQUIRED_MB_DEFAULT = MEASURED_PEAK_MIB + MEASURED_CUSHION_MIB
POLL_SEC_DEFAULT = 120
STABLE_POLLS_DEFAULT = 3


def sha256_file(path, chunk_bytes=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_hashes():
    return {p.name: sha256_file(p) for p in FROZEN if p.exists()}


def gpu_free_mb():
    """{gpu index: free MiB}, or {} if nvidia-smi is unavailable."""
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False)
    if p.returncode != 0:
        logger.warning("nvidia-smi failed (%s); treating as no capacity",
                       p.stderr.strip()[:200])
        return {}
    out = {}
    for line in p.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = int(parts[1])
    return out


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(DATASET_ROOT),
                          capture_output=True, text=True,
                          check=False).stdout.strip()


def git_dirty_tracked():
    p = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                       cwd=str(DATASET_ROOT), capture_output=True, text=True,
                       check=False)
    return [l for l in p.stdout.splitlines() if l.strip()]


def run_check(label, argv):
    """Run a verifier in its OWN process and report its exit status.

    A subprocess rather than an import: the verifier modules are the authority
    on their own artifacts, and importing them here would let this script's own
    module cache decide the answer instead of the committed code.
    """
    p = subprocess.run(argv, cwd=str(DATASET_ROOT), capture_output=True,
                       text=True, check=False,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                            "HOME": str(Path.home()),
                            "PYTHONDONTWRITEBYTECODE": "1"})
    ok = p.returncode == 0
    logger.info("%s %s (exit %d)", "PASS" if ok else "FAIL", label,
                p.returncode)
    if not ok:
        tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
        for line in tail:
            logger.error("  %s", line)
    return ok


def preflight(expect_commit=None):
    """Every condition that must hold at the moment of launch."""
    problems = []
    dirty = git_dirty_tracked()
    if dirty:
        problems.append(f"tracked worktree dirty: {dirty[:5]}")
    head = git_head()
    if expect_commit and not head.startswith(expect_commit):
        problems.append(f"HEAD is {head[:12]}, expected {expect_commit}")
    for path in FROZEN:
        if not path.exists():
            problems.append(f"frozen artifact missing: {path}")
    if problems:
        return {"ok": False, "problems": problems, "head": head}

    checks = [
        ("G6.0 hierarchy --verify --recount-source",
         [PY, "scripts/e2c_v3_mllmu_hierarchy.py", "--verify",
          "--recount-source"]),
        ("G6.1 pilot design --verify",
         [PY, "scripts/e2c_v3_mllmu_matrix.py", "--verify"]),
    ]
    for label, argv in checks:
        if not run_check(label, argv):
            problems.append(f"verification failed: {label}")
    return {"ok": not problems, "problems": problems, "head": head}


def launch_command(device):
    return [PY, "scripts/e2c_v3_granularity_matrix.py",
            "--dataset", "mllmu", "--phase", "all", "--device", device]


def wait_for_capacity(required_mb, poll_sec, stable_polls, deadline_sec=None,
                      exclude=()):
    """Block until one GPU has held >= required_mb for stable_polls in a row.

    Returns the device string ("cuda:N") or None if the deadline passed.  The
    consecutive-poll requirement is the whole point: a single sample can catch
    another job between allocations, and launching into that gap OOMs after the
    expensive part of the run has already been paid for.
    """
    streak = {}
    started = time.time()
    while True:
        free = gpu_free_mb()
        eligible = {i: mb for i, mb in free.items()
                    if mb >= required_mb and i not in exclude}
        for idx in free:
            streak[idx] = streak.get(idx, 0) + 1 if idx in eligible else 0
        best = max(free.items(), key=lambda kv: kv[1]) if free else None
        logger.info(
            "capacity poll: free MiB %s (need %d on one GPU, %d/%d stable "
            "polls%s)",
            {i: mb for i, mb in sorted(free.items())}, required_mb,
            max(streak.values()) if streak else 0, stable_polls,
            f", best {best[1]} on GPU {best[0]}" if best else "")
        ready = [i for i in eligible if streak[i] >= stable_polls]
        if ready:
            # Most free memory, not lowest index.  Every GPU in ``ready`` has
            # cleared the threshold, but headroom is not free: activation
            # growth over 15 cells is not perfectly predictable, and the
            # difference between launching on the tightest qualifying GPU and
            # the roomiest one is the difference between a run that finishes
            # and one that OOMs after the 3000-step baseline.
            idx = max(ready, key=lambda i: free[i])
            logger.info("GPU %d has held %d MiB free for %d consecutive polls "
                        "(most headroom of the %d eligible)",
                        idx, free[idx], stable_polls, len(ready))
            return f"cuda:{idx}"
        if deadline_sec and time.time() - started > deadline_sec:
            logger.warning("deadline of %ds reached without capacity",
                           deadline_sec)
            return None
        time.sleep(poll_sec)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--required-mb", type=int, default=REQUIRED_MB_DEFAULT,
                   help=f"free MiB needed on ONE GPU "
                        f"(default {REQUIRED_MB_DEFAULT})")
    p.add_argument("--poll-sec", type=int, default=POLL_SEC_DEFAULT)
    p.add_argument("--stable-polls", type=int, default=STABLE_POLLS_DEFAULT,
                   help="consecutive polls a GPU must hold the memory before "
                        "it counts, so a transient gap does not trigger an OOM")
    p.add_argument("--deadline-sec", type=int, default=None,
                   help="give up after this long (default: wait indefinitely)")
    p.add_argument("--expect-commit", default=None,
                   help="refuse to launch unless HEAD starts with this")
    p.add_argument("--exclude-gpu", type=int, nargs="*", default=[],
                   help="GPU indices to leave alone (e.g. one reserved for "
                        "another job)")
    p.add_argument("--dry-run", action="store_true",
                   help="run every check and print the command, do not launch")
    p.add_argument("--check-now", action="store_true",
                   help="run the preflight once and exit, without waiting")
    args = p.parse_args(argv)

    before = frozen_hashes()
    logger.info("frozen artifacts at queue start: %s",
                json.dumps({k: v[:12] for k, v in before.items()}, indent=2))

    if args.check_now:
        res = preflight(args.expect_commit)
        print(json.dumps({"ok": res["ok"], "head": res["head"],
                          "problems": res["problems"],
                          "gpu_free_mb": gpu_free_mb(),
                          "required_mb": args.required_mb}, indent=2))
        return 0 if res["ok"] else 1

    logger.info("waiting for a GPU with >= %d MiB free, stable over %d polls "
                "of %ds", args.required_mb, args.stable_polls, args.poll_sec)
    device = wait_for_capacity(args.required_mb, args.poll_sec,
                               args.stable_polls, args.deadline_sec,
                               set(args.exclude_gpu))
    if device is None:
        return 2

    # Re-check at the moment of launch, not at the moment the queue started:
    # capacity may have taken hours to appear, and the tree, HEAD and the
    # frozen artifacts can all have moved in that time.
    res = preflight(args.expect_commit)
    if not res["ok"]:
        for prob in res["problems"]:
            logger.error("refusing to launch: %s", prob)
        return 1
    after = frozen_hashes()
    if before != after:
        moved = {k: (before.get(k), after.get(k))
                 for k in set(before) | set(after)
                 if before.get(k) != after.get(k)}
        logger.error(
            "refusing to launch: frozen manifest(s) changed while waiting for "
            "capacity: %s.  Once pilot execution begins these are evidence, "
            "not inputs to regenerate -- investigate before launching.",
            json.dumps(moved, indent=2))
        return 1

    cmd = launch_command(device)
    logger.info("preflight passed at HEAD %s; launching: %s",
                res["head"][:12], " ".join(cmd))
    if args.dry_run:
        print(json.dumps({"dry_run": True, "device": device,
                          "command": cmd, "head": res["head"],
                          "frozen_hashes": before}, indent=2))
        return 0
    return subprocess.run(cmd, cwd=str(DATASET_ROOT), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
