"""Commit everything under results/ and push it back to the repo.

The PC workflow is: run experiments locally (they write into results/), then run
this to version the outputs back to GitHub. Small text/figure artifacts only —
large binaries (checkpoints, the AirfRANS cache) are kept out by .gitignore, so
this never tries to push gigabytes.

    python scripts/push_results.py
    python scripts/push_results.py -m "full run, seeds 0-4, RTX 4070 Ti"
    python scripts/push_results.py --no-push        # commit locally only

It is a thin, transparent wrapper over git: it stages `results/`, commits with a
timestamped message (only if something changed), and pushes the current branch.
Run it from anywhere inside the repo.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args: str, check: bool = True, capture: bool = False) -> str:
    """Run a git command in the repo root; return stdout when captured."""
    res = subprocess.run(
        ["git", "-C", _ROOT, *args],
        text=True,
        capture_output=capture,
        check=False,
    )
    if check and res.returncode != 0:
        msg = (res.stderr or res.stdout or "").strip()
        raise SystemExit(f"git {' '.join(args)} failed: {msg}")
    return (res.stdout or "").strip() if capture else ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Commit + push results/ back to the repo.")
    p.add_argument("-m", "--message", default=None, help="commit message")
    p.add_argument("--no-push", action="store_true", help="commit locally; do not push")
    args = p.parse_args(argv)

    results_dir = os.path.join(_ROOT, "results")
    if not os.path.isdir(results_dir):
        print(f"No results/ directory at {results_dir} — nothing to push.")
        return 0

    branch = _git("rev-parse", "--abbrev-ref", "HEAD", capture=True)
    _git("add", "--", "results")

    # Anything staged under results/? (diff --cached is empty -> exit 0)
    staged = subprocess.run(
        ["git", "-C", _ROOT, "diff", "--cached", "--quiet", "--", "results"],
        check=False,
    ).returncode
    if staged == 0:
        print("results/ is already up to date with the last commit — nothing to do.")
        return 0

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    message = args.message or f"results: experiment outputs ({stamp})"
    _git("commit", "-m", message)
    print(f"committed on '{branch}': {message}")

    if args.no_push:
        print("--no-push set; skipping push. Run `git push` when ready.")
        return 0

    print(f"pushing '{branch}' to origin ...")
    _git("push", "origin", branch)
    print("done — results are on the remote.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
