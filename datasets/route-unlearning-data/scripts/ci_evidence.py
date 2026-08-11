#!/usr/bin/env python3
"""R18: record CI evidence for the current commit.

This script captures the exact commit SHA, test count, and timestamp so GPU
generation runs can verify CI passed on the exact commit being used. Run this
before any GPU generation and record the output in the run log.

Usage:
    python scripts/ci_evidence.py

The script exits with code 0 if all tests pass, non-zero otherwise. The output
includes the commit SHA so it can be cross-referenced with the GPU run manifest.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_commit_sha() -> str:
    """Return the current Git commit SHA (short form)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_commit_sha_full() -> str:
    """Return the current Git commit SHA (full form)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _run_tests() -> tuple[int, int]:
    """Run the full test suite and return (exit_code, test_count)."""
    repo_root = Path(__file__).resolve().parent.parent
    package_dir = repo_root / "datasets" / "route-unlearning-data"
    
    # Run pytest with --collect-only first to count tests
    try:
        collect_result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", 
             "tests/unit", "tests/golden", "tests/integration"],
            capture_output=True,
            text=True,
            cwd=package_dir,
            check=False,
        )
        # Parse test count from output (last line like "52 tests collected")
        test_count = 0
        for line in collect_result.stdout.splitlines():
            if "test" in line.lower() and any(c.isdigit() for c in line):
                # Extract number from lines like "52 passed" or "52 tests collected"
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        test_count = int(part)
                        break
    except Exception:
        test_count = 0
    
    # Run the actual tests
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", 
             "tests/unit", "tests/golden", "tests/integration"],
            capture_output=True,
            text=True,
            cwd=package_dir,
            check=False,
        )
        return result.returncode, test_count
    except Exception as e:
        print(f"ERROR: failed to run tests: {e}", file=sys.stderr)
        return 1, test_count


def main() -> int:
    """Record CI evidence and exit with test status."""
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_sha_short = _git_commit_sha()
    commit_sha_full = _git_commit_sha_full()
    
    print("=" * 72)
    print("CI EVIDENCE FOR PRODUCTION COMMIT")
    print("=" * 72)
    print(f"Timestamp (UTC):     {timestamp}")
    print(f"Commit SHA (short):  {commit_sha_short}")
    print(f"Commit SHA (full):   {commit_sha_full}")
    print()
    print("Running full test suite...")
    print()
    
    exit_code, test_count = _run_tests()
    
    if exit_code == 0:
        print()
        print("=" * 72)
        print("CI STATUS: PASS")
        print("=" * 72)
        print(f"Tests collected: {test_count}")
        print(f"Commit {commit_sha_short} is ready for GPU generation.")
        print()
        print("Record this output in the GPU run log to verify CI passed on")
        print("the exact commit used for generation (repair plan R18).")
        return 0
    else:
        print()
        print("=" * 72)
        print("CI STATUS: FAIL")
        print("=" * 72)
        print(f"Tests collected: {test_count}")
        print(f"Commit {commit_sha_short} has failing tests.")
        print()
        print("Do NOT proceed with GPU generation until all tests pass.")
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
