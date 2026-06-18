"""Live, read-only web dashboard for the NeuroForge full research run.

This server is **fully decoupled** from training: it never imports neuroforge,
torch or numpy, never touches the training loop, and adds *zero* runtime cost to
the run. It only

  * READS the existing log file (default ``full_run.log``) on each poll, and
  * shells out to ``nvidia-smi`` from this separate process (cached <=1.5 s).

There is no hook in the training code — the run is observed purely through the
artifacts it already produces. Controls (stop / pause / resume) act on the OS
process and on a small flag file; they do not reach into the training process'
memory.

Run it::

    .venv\\Scripts\\python.exe scripts\\dashboard\\server.py --open

Stdlib only — no Flask / FastAPI / Streamlit / Node / new dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # scripts/dashboard -> repo
_INDEX_HTML = os.path.join(_HERE, "index.html")

# The future orchestrator hook (added by a separate change) will look here.
_CONTROL_FILE = os.path.join(_REPO_ROOT, "run_control.json")

# The orchestrator entry point we manage. Matched against process command lines.
_RUN_SCRIPT = "run_full_research"

# Preset "full" shape — used by the progress heuristic. Kept in sync with
# scripts/run_full_research.py::_PRESETS["full"]. If the log ever advertises a
# different shape we adapt below (see parse_status).
_FULL = dict(
    seeds=3,
    epochs=80,            # backbone epochs
    corrector_epochs=20,  # corrector epochs (trained per DEQ arm)
    arms=3,               # trainings per seed that run a backbone (see below)
    ood_tasks=2,          # reynolds, aoa
)

# --------------------------------------------------------------------------- #
# Log parsing (pure function — unit-tested)
# --------------------------------------------------------------------------- #

# Each ablation seed runs three *backbone* trainings (the DEQ arm, the
# local-corrector arm, the no-physics-loss arm); the DEQ arm additionally trains
# a corrector. We count "arm units" as those three backbone trainings per seed.
_ARMS_PER_SEED = 3

_STAGE1_RE = re.compile(r"\[stage 1\]\s+in-distribution")
_STAGE2_RE = re.compile(r"\[stage 2\]\s+out-of-distribution")
_DONE_RE = re.compile(r"\[done\]\s+all stages complete")

# in-dist:  "[ablation] seed 0: DEQ-corrector arm ..."
_ABL_RE = re.compile(
    r"\[ablation\]\s+seed\s+(\d+):\s+([A-Za-z0-9\- ]+?)\s+arm", re.IGNORECASE
)
# OOD:      "[ood-ablation] reynolds | seed 0: DEQ-corrector arm ..."
_OOD_ABL_RE = re.compile(
    r"\[ood-ablation\]\s+(\S+)\s*\|\s*seed\s+(\d+):\s+([A-Za-z0-9\- ]+?)\s+arm",
    re.IGNORECASE,
)
_OOD_TASK_RE = re.compile(r"\[ood-ablation\]\s+OOD task '([^']+)'")

# "[epoch 12] train_loss=1.2345e-02 val_loss=3.4e-02"
_EPOCH_RE = re.compile(
    r"\[epoch\s+(\d+)\]\s+train_loss=([0-9.eE+\-]+)\s+val_loss=([0-9.eE+\-]+)"
)
# "[train] epoch 3 step 120 loss=1.2e-02 ..."  (intra-epoch step line)
_STEP_RE = re.compile(r"\[train\]\s+epoch\s+(\d+)\s+step\s+(\d+)\s+loss=([0-9.eE+\-]+)")
# "[corrector epoch 4] loss=1.2e-02 ..."
_CORR_RE = re.compile(r"\[corrector epoch\s+(\d+)\]\s+loss=([0-9.eE+\-]+)")

# Baseline run (scripts/run_baselines.py): a matched-budget Transolver backbone
# trained over --seeds 0 1 2, --epochs each, with NO corrector phase, e.g.:
#   "[baseline] seed 0 epoch 26: train_mse=3.9557e-02 (77.6s)"
_BASELINE_RE = re.compile(
    r"\[baseline\]\s+seed\s+(\d+)\s+epoch\s+(\d+):\s+train_mse=([0-9.eE+\-]+)"
)
# Final baseline line references the Table 2 artifacts, e.g.:
#   "[baseline] wrote results/baselines/table2.md and table2.csv"
# (also accept an explicit "baselines complete" marker if one is ever emitted).
_BASELINE_DONE_RE = re.compile(r"table2|baselines complete", re.IGNORECASE)

# NeuroForge v2 run (scripts/run_v2.py): a SOTA Transolver backbone trained on the
# native point cloud, then the certified trust layer (residual detector + conformal
# coverage) + learned corrector run ON its predictions, per seed. All lines carry
# the "[v2]" prefix, so they never clobber the ablation/baseline/single formats.
#   "[v2] seed 0: Transolver params = 7,350,420"
_V2_SEED_RE = re.compile(r"\[v2\]\s+seed\s+(\d+):\s+Transolver params")
#   "[v2] seed 0 backbone epoch 12: train_mse=3.9557e-02 (45.3s)"
_V2_BB_RE = re.compile(
    r"\[v2\]\s+seed\s+(\d+)\s+backbone epoch\s+(\d+):\s+train_mse=([0-9.eE+\-]+)"
    r"\s+\(([0-9.]+)s\)"
)
#   "[v2] seed 0 corrector epoch 4: loss=1.2e-02"
_V2_CORR_RE = re.compile(
    r"\[v2\]\s+seed\s+(\d+)\s+corrector epoch\s+(\d+):\s+loss=([0-9.eE+\-]+)"
)
_V2_EVAL_A_RE = re.compile(r"\[v2\]\s+seed\s+(\d+):\s+evaluating \(a\)")
_V2_EVAL_B_RE = re.compile(r"\[v2\]\s+seed\s+(\d+):\s+evaluating \(b")
_V2_CORR_TRAIN_RE = re.compile(
    r"\[v2\]\s+seed\s+(\d+):\s+(?:training|pre-computing)"
)
_V2_CONF_RE = re.compile(r"\[v2\]\s+seed\s+(\d+):\s+split-conformal")
# Headline result lines, e.g.:
#   "[v2] seed 0: (a) eval 12.3s -> mse_u=0.12, ..., residual_error_spearman=0.41"
_V2_RESULT_RE = re.compile(
    r"\[v2\]\s+seed\s+\d+:\s+\((a|b/[A-Za-z]+)\)\s+eval[^>]*->\s+(.*)$"
)
_V2_SPEARMAN_RE = re.compile(r"residual_error_spearman=([0-9.eE+\-]+)")
_V2_CONF_RESULT_RE = re.compile(r"\[v2\]\s+seed\s+\d+:\s+conformal\s+->\s+(.*)$")
# The very last line of a finished run: "[v2] artifacts in results/v2".
_V2_DONE_RE = re.compile(r"\[v2\]\s+artifacts in")

# tqdm: "rasterise airfrans/full/train:  30%|###| 242/800 [22:56<1:04:22,  6.92s/it]"
#       "Loading dataset (task: full, split: train):  30%|..| 240/800 [..]"
_TQDM_RE = re.compile(
    r"^(?P<desc>.*?):\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
)


def _to_float(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_status(log_text: str) -> dict:
    """Parse the raw research log into a structured run-state dict.

    Robust to short / empty logs and to tqdm carriage-return segments: the text
    is split on both ``\\r`` and ``\\n`` and the *latest* matching segment wins.
    Every field defaults cleanly so the dashboard renders a graceful "waiting"
    state before any real output exists.

    Progress heuristic (smooth + monotone, not exact)
    -------------------------------------------------
    The bar is composed of two top-level stages, each weighted by how many
    backbone trainings it contains:

        stage 1 (in-dist) : seeds * arms_per_seed          = 3 * 3 = 9 units
        stage 2 (ood)     : ood_tasks * seeds * arms        = 2 * 3 * 3 = 18 units
        total                                                 = 27 units

    Within a stage the fraction is::

        (completed_units + current_unit_fraction) / total_units

    where ``current_unit_fraction`` blends the data-prep phase and the training
    phase of the unit currently in flight:

      * before any epoch line for the current arm we are *preparing data*; we
        use the tqdm fraction (rasterise / loading) scaled into the first 20% of
        the unit, so the bar moves during the long ~40-min rasterise instead of
        sitting at 0;
      * once epochs start, the remaining 80% of the unit tracks
        (epoch+1)/total_epochs over the backbone epochs.

    Only the *first* unit of a stage actually rasterises (the dataset is cached
    after that), so the data-prep weight applies mostly to that first unit; for
    later units with no fresh tqdm the fraction simply starts from the epoch
    progress. The result is always in [0, 100] and only ever increases as the
    run advances through stages, seeds, arms and epochs.
    """
    # Split on CR or LF so tqdm in-place updates become individual segments.
    segments = re.split(r"[\r\n]+", log_text or "")
    segments = [s for s in segments if s.strip()]

    state = {
        "stage": None,            # "in_dist" | "ood" | "done" | None
        "stage_label": None,      # human label
        "ood_task": None,         # current OOD task name (reynolds/aoa) if any
        "seed": None,             # current seed index
        "seeds_total": _FULL["seeds"],
        "arm": None,              # current arm name
        "epoch": None,            # latest backbone epoch (0-indexed in log)
        "epochs_total": _FULL["epochs"],
        "corrector_epoch": None,
        "corrector_epochs_total": _FULL["corrector_epochs"],
        "train_loss": None,
        "val_loss": None,
        "step_loss": None,
        "loss_history": [],       # recent train_loss values for the sparkline
        "dataprep": None,         # {desc, cur, tot, pct} or None
        "phase": "waiting",       # waiting | preparing | training | done | stopped
        "progress": 0.0,          # 0..100
        "last_segment": segments[-1].strip() if segments else None,
        # --- v2 run extras (Transolver backbone + certified trust layer) --- #
        "v2_phase": None,         # backbone | eval-a | corrector | eval-b | conformal
        "sec_per_epoch": None,    # latest logged backbone wall-time (for ETA)
        "eta_seconds": None,      # estimated time remaining (v2; backbone-dominated)
        "v2_spearman": None,      # latest residual<->error Spearman (the headline)
        "v2_result_line": None,   # latest raw eval result string (for the UI)
        "v2_conformal": None,     # latest conformal result string
        "phase_elapsed_s": None,  # seconds in the current log-silent phase (creep)
    }

    # ---- single forward pass; latest match wins for each field ---- #
    saw_any_epoch_for_current_unit = False
    last_dataprep = None

    for seg in segments:
        if _DONE_RE.search(seg):
            state["stage"] = "done"
            state["stage_label"] = "All stages complete"
            state["phase"] = "done"
            continue

        # ---- NeuroForge v2 (Transolver backbone + certified trust layer). All
        # lines start with "[v2]"; we track the per-seed sub-phase, capture the
        # backbone wall-time for an ETA, and surface the headline residual<->error
        # Spearman as it forms. Distinct prefix => never clobbers other modes.
        if seg.lstrip().startswith("[v2]"):
            if _V2_DONE_RE.search(seg):
                state["stage"] = "v2"
                state["stage_label"] = "NeuroForge v2 run complete"
                state["phase"] = "done"
                continue
            mb = _V2_BB_RE.search(seg)
            if mb:
                state["stage"] = "v2"
                state["seed"] = int(mb.group(1))
                state["epoch"] = int(mb.group(2))
                tl = _to_float(mb.group(3))
                state["train_loss"] = tl
                if tl is not None:
                    state["loss_history"].append(tl)
                spe = _to_float(mb.group(4))
                if spe is not None:
                    state["sec_per_epoch"] = spe
                state["v2_phase"] = "backbone"
                saw_any_epoch_for_current_unit = True
                last_dataprep = None
                continue
            mc = _V2_CORR_RE.search(seg)
            if mc:
                state["stage"] = "v2"
                state["seed"] = int(mc.group(1))
                state["corrector_epoch"] = int(mc.group(2))
                state["step_loss"] = _to_float(mc.group(3))
                state["v2_phase"] = "corrector"
                last_dataprep = None
                continue
            ms = _V2_SEED_RE.search(seg)
            if ms:
                # A new seed begins: reset the per-seed epoch trackers so the
                # previous seed's corrector epoch can't leak into this seed.
                state["stage"] = "v2"
                state["seed"] = int(ms.group(1))
                state["epoch"] = None
                state["corrector_epoch"] = None
                state["v2_phase"] = "backbone"
                saw_any_epoch_for_current_unit = False
                continue
            mr = _V2_RESULT_RE.search(seg)
            if mr:
                state["stage"] = "v2"
                state["v2_result_line"] = mr.group(2).strip()
                msp = _V2_SPEARMAN_RE.search(mr.group(2))
                if msp:
                    state["v2_spearman"] = _to_float(msp.group(1))
                continue
            mcr = _V2_CONF_RESULT_RE.search(seg)
            if mcr:
                state["stage"] = "v2"
                state["v2_conformal"] = mcr.group(1).strip()
                continue
            if _V2_EVAL_A_RE.search(seg):
                state["stage"] = "v2"; state["v2_phase"] = "eval-a"; continue
            if _V2_EVAL_B_RE.search(seg):
                state["stage"] = "v2"; state["v2_phase"] = "eval-b"; continue
            if _V2_CONF_RE.search(seg):
                state["stage"] = "v2"; state["v2_phase"] = "conformal"; continue
            if _V2_CORR_TRAIN_RE.search(seg):
                state["stage"] = "v2"; state["v2_phase"] = "corrector"; continue
            if state["stage"] is None:
                state["stage"] = "v2"
                state["stage_label"] = "NeuroForge v2 (Transolver + trust layer)"
            continue

        # ---- Baseline run (Transolver, seeds x backbone epochs, no corrector).
        # These lines all start with "[baseline]" and are distinct from the
        # ablation/single-run formats, so they never clobber that parsing.
        if seg.lstrip().startswith("[baseline]"):
            mb = _BASELINE_RE.search(seg)
            if mb:
                state["stage"] = "baseline"
                state["seed"] = int(mb.group(1))
                state["epoch"] = int(mb.group(2))
                tl = _to_float(mb.group(3))
                state["train_loss"] = tl
                if tl is not None:
                    state["loss_history"].append(tl)
                saw_any_epoch_for_current_unit = True
                last_dataprep = None
                state["stage_label"] = (
                    f"Baseline (Transolver) — seed {state['seed']}, "
                    f"epoch {state['epoch']}"
                )
                continue
            if _BASELINE_DONE_RE.search(seg):
                # table2 is written after EVERY seed (incremental save), so only
                # treat it as completion once the LAST seed's results are in —
                # otherwise the bar jumps to "done" after seed 0.
                if state["seed"] is not None and state["seed"] >= (_FULL["seeds"] - 1):
                    state["stage"] = "done"
                    state["stage_label"] = "Baseline run complete"
                    state["phase"] = "done"
                continue
            # Any other "[baseline]" line (setup/eval): mark the stage so the
            # dashboard labels it as a baseline run rather than a single run.
            if state["stage"] is None:
                state["stage"] = "baseline"
                state["stage_label"] = "Baseline (Transolver)"
            continue

        if _STAGE1_RE.search(seg):
            state["stage"] = "in_dist"
            state["stage_label"] = "Stage 1 — in-distribution ablation"
            state["ood_task"] = None
            continue
        if _STAGE2_RE.search(seg):
            state["stage"] = "ood"
            state["stage_label"] = "Stage 2 — out-of-distribution ablation"
            continue

        m = _OOD_TASK_RE.search(seg)
        if m:
            state["stage"] = "ood"
            state["ood_task"] = m.group(1)
            state["stage_label"] = f"Stage 2 — OOD: {m.group(1)}"
            continue

        # OOD arm line carries the task too.
        m = _OOD_ABL_RE.search(seg)
        if m:
            state["stage"] = "ood"
            state["ood_task"] = m.group(1)
            state["seed"] = int(m.group(2))
            state["arm"] = m.group(3).strip()
            state["stage_label"] = f"Stage 2 — OOD: {m.group(1)}"
            # new arm: reset per-unit epoch tracking
            state["epoch"] = None
            state["corrector_epoch"] = None
            saw_any_epoch_for_current_unit = False
            continue

        m = _ABL_RE.search(seg)
        if m:
            state["seed"] = int(m.group(1))
            state["arm"] = m.group(2).strip()
            if state["stage"] is None:
                state["stage"] = "in_dist"
                state["stage_label"] = "Stage 1 — in-distribution ablation"
            state["epoch"] = None
            state["corrector_epoch"] = None
            saw_any_epoch_for_current_unit = False
            continue

        m = _EPOCH_RE.search(seg)
        if m:
            state["epoch"] = int(m.group(1))
            tl = _to_float(m.group(2))
            vl = _to_float(m.group(3))
            state["train_loss"] = tl
            state["val_loss"] = vl
            if tl is not None:
                state["loss_history"].append(tl)
            saw_any_epoch_for_current_unit = True
            last_dataprep = None  # training has started; prep bar is no longer live
            continue

        m = _CORR_RE.search(seg)
        if m:
            state["corrector_epoch"] = int(m.group(1))
            cl = _to_float(m.group(2))
            if cl is not None:
                state["step_loss"] = cl
            saw_any_epoch_for_current_unit = True
            last_dataprep = None
            continue

        m = _STEP_RE.search(seg)
        if m:
            state["step_loss"] = _to_float(m.group(3))
            continue

        m = _TQDM_RE.match(seg)
        if m:
            last_dataprep = {
                "desc": m.group("desc").strip(),
                "cur": int(m.group("cur")),
                "tot": int(m.group("tot")),
                "pct": int(m.group("pct")),
            }
            continue

    # Keep the dataprep info around (it is what is actively scrolling now).
    state["dataprep"] = last_dataprep

    # Bound the loss history sent to the client. A long multi-seed run logs
    # hundreds of epoch losses; an unbounded array bloats every poll and the
    # client sparkline. Downsample EVENLY to <=240 points so the overall
    # convergence shape is preserved without growing without limit.
    _lh = state["loss_history"]
    if len(_lh) > 240:
        _step = len(_lh) / 240.0
        state["loss_history"] = [_lh[min(int(i * _step), len(_lh) - 1)] for i in range(240)]

    # ---- derive phase ---- #
    if state["phase"] != "done":
        if state["epoch"] is not None or state["corrector_epoch"] is not None:
            state["phase"] = "training"
        elif last_dataprep is not None:
            state["phase"] = "preparing"
        elif state["stage"] is not None:
            state["phase"] = "preparing"
        else:
            state["phase"] = "waiting"

    # Single-run (no ablation "[stage]" markers): label the cards sensibly so a
    # certificate/baseline training doesn't show blank stage/seed/arm fields.
    if state["stage"] is None and state["phase"] != "waiting":
        if state["corrector_epoch"] is not None:
            state["stage_label"] = "Single run — corrector training"
        elif state["epoch"] is not None:
            state["stage_label"] = "Single run — backbone training"
        elif last_dataprep is not None:
            state["stage_label"] = "Single run — preparing data"
        else:
            state["stage_label"] = "Single run"

    # ---- v2 run: human label, sub-phase as the "arm" card, and an ETA. ---- #
    if state["stage"] == "v2" and state["phase"] != "done":
        ph = state["v2_phase"] or "backbone"
        _ph_label = {
            "backbone": "Transolver backbone",
            "eval-a": "eval — backbone alone",
            "corrector": "training corrector (on residuals)",
            "eval-b": "eval — backbone + certified loop",
            "conformal": "split-conformal coverage",
        }.get(ph, ph)
        state["arm"] = _ph_label
        sd = state["seed"] if state["seed"] is not None else 0
        state["stage_label"] = (
            f"NeuroForge v2 — Transolver + trust layer  ·  seed {sd}/"
            f"{_FULL['seeds'] - 1}  ·  {_ph_label}"
        )
        state["eta_seconds"] = _v2_eta(state)

    state["progress"] = _compute_progress(
        state, saw_any_epoch_for_current_unit, last_dataprep
    )
    return state


def apply_v2_conformal_creep(status, now, log_mtime):
    """Advance the bar through the long, log-silent v2 conformal phase.

    The conformal MC-dropout pass writes nothing for ~2.5h, so the parser sits
    frozen. We creep the bar across the conformal slice (0.48..0.99 of the seed)
    from the wall-clock, anchored to ``log_mtime`` — the time the "split-conformal
    coverage ..." line was written, which IS when the phase started. Using mtime
    (not a fresh server timer) means a dashboard restart mid-conformal resumes at
    the right place instead of snapping backwards. Monotone: only ever raises
    ``progress`` (via max), and the slice end (0.99) is below the next seed's
    backbone start. Mutates and returns ``status``.
    """
    if status.get("stage") != "v2" or status.get("phase") == "done":
        return status
    if status.get("v2_phase") != "conformal" or not log_mtime:
        return status
    elapsed = max(0.0, now - log_mtime)
    frac = min(elapsed / _V2_CONFORMAL_EST_SECS, 0.98)
    seeds = _FULL["seeds"] or 1
    seed = status.get("seed") or 0
    wf = 0.48 + 0.51 * frac          # conformal occupies the 0.48..0.99 tail
    crept = min(100.0 * (seed + wf) / seeds, 99.5)
    status["progress"] = max(status.get("progress", 0.0), crept)
    status["phase_elapsed_s"] = elapsed
    return status


def _v2_eta(state) -> float | None:
    """Rough time-remaining (s) for a v2 run, dominated by backbone epochs.

    Uses the latest logged backbone wall-time/epoch and counts the backbone
    epochs still to run across the remaining seeds, plus a flat corrector+eval
    allowance per remaining seed. Returns None until a backbone epoch is timed.
    """
    spe = state.get("sec_per_epoch")
    if not spe or spe <= 0:
        return None
    seeds = _FULL["seeds"] or 1
    bb = max(SINGLE_BB_EPOCHS, 1)
    seed = state["seed"] if state["seed"] is not None else 0
    epoch = state["epoch"] if state["epoch"] is not None else -1
    # Backbone epochs left in the current seed (0 once we leave the backbone phase).
    if state.get("v2_phase") == "backbone":
        bb_left_here = max(bb - (epoch + 1), 0)
    else:
        bb_left_here = 0
    bb_left_future = max(seeds - seed - 1, 0) * bb
    bb_secs = (bb_left_here + bb_left_future) * spe
    # Corrector + the two eval passes per seed: ~10% of a backbone's wall-time.
    ph = state.get("v2_phase")
    seeds_with_tail_left = max(seeds - seed - 1, 0) + (
        1 if ph in ("backbone", "eval-a", None) else 0
    )
    tail_secs = seeds_with_tail_left * 0.10 * bb * spe
    # The conformal MC pass is CPU-bound (~_V2_CONFORMAL_EST_SECS) and the single
    # biggest remaining cost — count it explicitly per seed whose conformal is
    # still ahead, else the ETA badly under-reports (backbone alone misses ~half).
    conf_left = max(seeds - seed - 1, 0)        # future seeds' conformal
    if ph == "conformal":
        conf_left += 0.5                        # current one ~half done (rough)
    elif ph in ("backbone", "eval-a", "corrector", "eval-b", None):
        conf_left += 1                          # current seed's conformal not started
    conf_secs = conf_left * _V2_CONFORMAL_EST_SECS
    return float(bb_secs + tail_secs + conf_secs)


# The one-time initial AirfRANS rasterise/load takes ~40 min (the dataset is
# cached afterwards). To avoid the bar sitting at ~0 for that whole phase, the
# first PREP_SLICE of the bar is dedicated to that initial data-prep, driven
# directly by the tqdm fraction; training then occupies the remaining
# (1 - PREP_SLICE). Smooth and strictly monotone across the handoff.
_PREP_SLICE = 0.05  # 5% of the overall bar reserved for the initial data load

# --- single-run mode: a standalone training (certificate / baseline runs) with
# no ablation "[stage]" markers. The bar is driven by epoch progress instead of
# the 27-unit ablation model: backbone fills 0..70%, corrector 70..100%, and it
# holds ~99% during post-training measurement. Budgets set via --bb/--corr-epochs.
SINGLE_BB_EPOCHS = 40
SINGLE_CORR_EPOCHS = 15

# The v2 split-conformal step (MC-dropout + rasterise per case) is CPU-bound and
# emits NO log lines for its whole duration (~2.5h/seed measured), so the pure
# parser alone would freeze the bar. apply_v2_conformal_creep() advances it from
# the wall-clock, anchored to the log's last-write time (= when "split-conformal
# coverage ..." was printed, i.e. when the phase began). Tune if the box differs.
_V2_CONFORMAL_EST_SECS = 9000.0  # ~2.5 h

# Substring identifying the run process for Stop / alive-detection. Set from
# --run-match so the dashboard can watch run_full_research, run_certificates, etc.
RUN_MATCH = "run_full_research"


def _compute_progress(state, saw_epoch_for_unit, dataprep) -> float:
    """Smooth, monotone 0..100 progress from the parsed state (see docstring)."""
    if state["phase"] == "done":
        return 100.0

    # ---- Baseline run (Transolver): seeds x backbone epochs, no corrector. ----
    # Linear over the full (seeds * bb_epochs) grid. Held in [1, 99.5] until the
    # process is done (the table2/complete line drives phase == "done" -> 100).
    if state["stage"] == "baseline":
        seeds = _FULL["seeds"] or 1
        bb = max(SINGLE_BB_EPOCHS, 1)
        seed = state["seed"] if state["seed"] is not None else 0
        epoch = state["epoch"] if state["epoch"] is not None else 0
        frac = (seed * bb + (epoch + 1)) / float(seeds * bb)
        return max(1.0, min(100.0 * frac, 99.5))

    # ---- NeuroForge v2 run: per-seed phases (backbone -> corrector -> evals). --
    # Each seed's slice is split: backbone 0..60%, corrector 60..85%, evals +
    # conformal 85..100% of that seed's share. Strictly monotone, held < 99.5
    # until the "artifacts in" line drives phase == "done" -> 100.
    if state["stage"] == "v2":
        seeds = _FULL["seeds"] or 1
        bb = max(SINGLE_BB_EPOCHS, 1)
        ce = max(SINGLE_CORR_EPOCHS, 1)
        seed = state["seed"] if state["seed"] is not None else 0
        # Per-seed sub-phase fractions, weighted by REAL wall-time (not epoch
        # count) so the bar tracks elapsed time, and kept strictly ordered so it
        # is monotone across backbone -> eval-a -> corrector -> eval-b ->
        # conformal. Measured per-seed wall-times (~80s/backbone-epoch): backbone
        # ~1.8h, evals ~0.4h, corrector ~0.15h, conformal ~2.5h (CPU-bound). So
        # conformal alone is ~half the seed and gets the 0.48..0.99 tail; it emits
        # NO log lines, so its bar is crept from the wall-clock in
        # apply_v2_conformal_creep (here it just pins the slice start).
        ph = state.get("v2_phase")
        if ph == "conformal":
            wf = 0.48
        elif ph == "eval-b":
            wf = 0.47
        elif state["corrector_epoch"] is not None or ph == "corrector":
            cep = state["corrector_epoch"]
            wf = 0.40 + 0.05 * ((cep + 1) / ce if cep is not None else 0.0)
        elif ph == "eval-a":
            wf = 0.39
        elif state["epoch"] is not None:
            wf = 0.37 * ((state["epoch"] + 1) / bb)
        else:
            wf = 0.0
        overall = (seed + min(wf, 1.0)) / seeds
        return max(1.0, min(100.0 * overall, 99.5))

    if state["stage"] is None:
        return _single_run_progress(state)

    seeds = _FULL["seeds"] or 1
    arms = _ARMS_PER_SEED
    units_in1 = seeds * arms                    # 9
    units_in2 = _FULL["ood_tasks"] * seeds * arms  # 18
    total_units = units_in1 + units_in2         # 27

    # ---- Phase A: initial data-prep, before any epoch has ever appeared ---- #
    # While we are still rasterising/loading the very first arm of stage 1 (no
    # epoch line seen yet), animate the dedicated front slice from the tqdm bar.
    if not saw_epoch_for_unit and state["stage"] == "in_dist" and (
        state["seed"] in (None, 0)
    ) and dataprep is not None and dataprep["tot"] > 0:
        prep_frac = min(dataprep["cur"] / dataprep["tot"], 1.0)
        return 100.0 * _PREP_SLICE * prep_frac

    # ---- Phase B: training (data ready). Map arm-units into the rest. ---- #
    completed = 0
    if state["stage"] == "ood":
        completed += units_in1  # all of stage 1 finished
        # Account for earlier OOD tasks: each completed OOD task is seeds*arms
        # units. Without this, the 2nd task (aoa) is indistinguishable from the
        # 1st (reynolds) and the bar freezes. Order matches run_full_research's
        # default --ood-tasks (reynolds, aoa).
        _ood_order = {"reynolds": 0, "aoa": 1}
        task_idx = _ood_order.get((state["ood_task"] or "").lower(), 0)
        completed += task_idx * (seeds * arms)

    seed = state["seed"] if state["seed"] is not None else 0
    completed += seed * arms

    # Position of the current arm within its seed (0,1,2). Map known arm names;
    # unknown -> 0 (conservative, keeps monotonicity reasonable).
    arm_order = {
        "deq-corrector": 0,
        "local-corrector": 1,
        "no-physics-loss": 2,
    }
    arm_key = (state["arm"] or "").lower()
    arm_idx = arm_order.get(arm_key, 0)
    completed += arm_idx
    completed = min(completed, total_units - 1)

    # Fraction of the current unit done from its epoch progress; if a later unit
    # is itself rasterising (rare — cache usually hits) use the tqdm fraction.
    if saw_epoch_for_unit and state["epoch"] is not None:
        unit_frac = (state["epoch"] + 1) / max(_FULL["epochs"], 1)
    elif dataprep is not None and dataprep["tot"] > 0:
        unit_frac = 0.0  # mid prep of a later unit; hold at the unit boundary
    else:
        unit_frac = 0.0
    unit_frac = min(unit_frac, 1.0)

    # Training occupies the bar from _PREP_SLICE..1.0.
    train_frac = (completed + unit_frac) / total_units
    progress = 100.0 * (_PREP_SLICE + (1.0 - _PREP_SLICE) * train_frac)
    return max(0.0, min(progress, 100.0))


def _single_run_progress(state) -> float:
    """Epoch-based progress for a standalone training (no ablation stages).

    Backbone epochs fill 0..70% of the bar, corrector epochs 70..100%; holds at
    ~99% during the post-training measurement phase until the process exits.
    """
    if state["phase"] == "done":
        return 100.0
    ce = state["corrector_epoch"]
    if ce is not None:
        frac = (ce + 1) / max(SINGLE_CORR_EPOCHS, 1)
        return max(70.0, min(70.0 + 30.0 * frac, 99.0))
    ep = state["epoch"]
    if ep is not None:
        frac = (ep + 1) / max(SINGLE_BB_EPOCHS, 1)
        return max(2.0, min(70.0 * frac, 70.0))
    dp = state.get("dataprep")
    if dp and dp.get("tot"):
        return min(2.0 * (dp["cur"] / dp["tot"]), 2.0)
    return 0.0


# --------------------------------------------------------------------------- #
# GPU polling (cached) — separate process, no coupling to training
# --------------------------------------------------------------------------- #

_GPU_QUERY = (
    "temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw"
)
_gpu_cache = {"ts": 0.0, "data": None}
_gpu_lock = threading.Lock()
_GPU_TTL = 1.5  # seconds — rapid polling stays cheap


def _read_gpu_uncached():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_GPU_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    # First GPU only.
    line = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None

    return {
        "temp_c": num(parts[0]),
        "util_pct": num(parts[1]),
        "mem_used_mb": num(parts[2]),
        "mem_total_mb": num(parts[3]),
        "power_w": num(parts[4]),
    }


def get_gpu():
    """Return GPU telemetry, cached to at most once per ``_GPU_TTL`` seconds."""
    now = time.time()
    with _gpu_lock:
        if _gpu_cache["data"] is not None and (now - _gpu_cache["ts"]) < _GPU_TTL:
            return _gpu_cache["data"]
    data = _read_gpu_uncached()
    with _gpu_lock:
        _gpu_cache["ts"] = time.time()
        _gpu_cache["data"] = data
    return data


# --------------------------------------------------------------------------- #
# System (CPU/RAM) polling (cached) — one PowerShell spawn, decoupled like GPU
# --------------------------------------------------------------------------- #

# A PowerShell cold-start + CIM enumeration is heavier than nvidia-smi, so this
# uses a slightly longer TTL than the GPU read. Per-second polling therefore
# spawns PowerShell at most once per ~2.5 s — keeps it cheap and decoupled from
# the training job.
_SYS_CACHE = {"ts": 0.0, "data": None}
_sys_lock = threading.Lock()
_SYS_TTL = 2.5  # seconds — slightly longer than _GPU_TTL (PowerShell is heavier)

# One single subprocess call returns "cpuLoad,totalKB,freeKB".
_SYS_PS_CMD = (
    "$c=(Get-CimInstance Win32_Processor | "
    "Measure-Object -Property LoadPercentage -Average).Average; "
    "$o=Get-CimInstance Win32_OperatingSystem; "
    "\"$c,$($o.TotalVisibleMemorySize),$($o.FreePhysicalMemory)\""
)


def _read_sys_uncached():
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _SYS_PS_CMD,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    line = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        cpu = float(parts[0])
        total_kb = float(parts[1])
        free_kb = float(parts[2])
    except ValueError:
        return None
    return {
        "cpu_pct": cpu,
        "ram_used_mb": (total_kb - free_kb) / 1024.0,
        "ram_total_mb": total_kb / 1024.0,
    }


def get_sys():
    """Return CPU/RAM telemetry, cached to at most once per ``_SYS_TTL`` seconds."""
    now = time.time()
    with _sys_lock:
        if _SYS_CACHE["data"] is not None and (now - _SYS_CACHE["ts"]) < _SYS_TTL:
            return _SYS_CACHE["data"]
    data = _read_sys_uncached()
    with _sys_lock:
        _SYS_CACHE["ts"] = time.time()
        _SYS_CACHE["data"] = data
    return data


# --------------------------------------------------------------------------- #
# Process control (Stop) — OS-level, no coupling to training memory
# --------------------------------------------------------------------------- #

# The process lookup shells out to PowerShell/wmic, which is far too expensive to
# do on every 1 s status poll (a powershell cold-start + full CIM enumeration per
# second would contend with training — exactly what "zero impact" forbids). So we
# cache it like the GPU read. ``stop_run`` always forces a fresh lookup so it acts
# on current truth.
_PROC_TTL = 2.5  # seconds
_proc_cache = {"ts": 0.0, "data": None}
_proc_lock = threading.Lock()


def find_run_pids(force=False):
    """Cached wrapper around :func:`_find_run_pids` (see ``_PROC_TTL``)."""
    now = time.time()
    if not force:
        with _proc_lock:
            if _proc_cache["data"] is not None and (now - _proc_cache["ts"]) < _PROC_TTL:
                return _proc_cache["data"]
    pids = _find_run_pids()
    with _proc_lock:
        _proc_cache["ts"] = time.time()
        _proc_cache["data"] = pids
    return pids


def _find_run_pids():
    """Return PIDs whose command line references the run orchestrator.

    Prefers PowerShell ``Get-CimInstance`` (always present on modern Windows;
    ``wmic`` is an *optional* feature on Win11 24H2+ and may be missing), and
    falls back to ``wmic`` when CIM is unavailable. We never add psutil.

    Self-matches (this server, the CIM/wmic query itself, the shell running it)
    are filtered out so Stop only targets the real training job.
    """
    me = os.getpid()
    pids = []

    # ---- primary: CIM via PowerShell ---- #
    ps_cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match '" + RUN_MATCH + "' } | "
        "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                pid_s, cmd = line.split("\t", 1)
                pids.extend(_keep_pid(pid_s, cmd, me))
            if pids:
                return sorted(set(pids))
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # ---- fallback: wmic ---- #
    try:
        out = subprocess.run(
            ["wmic", "process", "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if RUN_MATCH not in line:
                    continue
                # csv: Node,CommandLine,ProcessId
                cells = line.split(",")
                if len(cells) < 3:
                    continue
                pid_s = cells[-1].strip()
                cmd = ",".join(cells[1:-1])
                pids.extend(_keep_pid(pid_s, cmd, me))
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return sorted(set(pids))


def _keep_pid(pid_s, cmd, me):
    """Decide whether a matched command line is the real training job.

    The real run is always a *Python interpreter* executing the orchestrator
    script, e.g. ``...\\python.exe scripts/run_full_research.py --preset full``.
    We accept only that shape and reject everything else that merely *mentions*
    ``run_full_research`` — the CIM/wmic query itself, a shell (bash/cmd/
    powershell) that has the name on its command line, an editor, ``taskkill``,
    or this dashboard server. This keeps ``taskkill /T`` from ever tree-killing a
    bystander (e.g. the very shell that launched the lookup).

    Returns ``[pid]`` to keep or ``[]`` to drop.
    """
    try:
        pid = int(pid_s)
    except ValueError:
        return []
    if pid == me:
        return []
    low = cmd.lower()

    # Anything that is *searching for* the script rather than *running* it.
    if "get-ciminstance" in low or "win32_process" in low:
        return []
    if "wmic" in low and "process" in low:
        return []
    if "taskkill" in low or "tasklist" in low:
        return []
    if "server.py" in low and "dashboard" in low:
        return []  # this dashboard server itself

    # Must be a Python interpreter actually invoking the orchestrator .py file.
    is_python = ("python" in low) or ("\\python" in low) or low.startswith("py ")
    runs_script = (RUN_MATCH in low) and (".py" in low)
    if not (is_python and runs_script):
        return []
    return [pid]


def stop_run():
    pids = find_run_pids(force=True)  # act on current truth, not a stale cache
    if not pids:
        return {"ok": False, "message": "No running run_full_research process found."}
    killed = []
    errors = []
    for pid in pids:
        try:
            out = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0:
                killed.append(pid)
            else:
                errors.append(f"pid {pid}: {out.stderr.strip() or out.stdout.strip()}")
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            errors.append(f"pid {pid}: {exc}")
    if killed and not errors:
        return {"ok": True, "message": f"Stopped run (PID {', '.join(map(str, killed))})."}
    if killed:
        return {"ok": True, "message": f"Stopped PID {killed}; issues: {'; '.join(errors)}"}
    return {"ok": False, "message": "; ".join(errors) or "taskkill failed."}


def set_paused(paused: bool):
    """Write/remove the pause flag file.

    NOTE: the training orchestrator does not yet consume this flag — a separate
    change will add a hook in run_full_research.py that checks ``run_control.json``
    at stage/seed boundaries and blocks while ``paused`` is true. For now this
    only manages the flag file so the wiring is ready and the UI is functional.
    """
    try:
        if paused:
            with open(_CONTROL_FILE, "w", encoding="utf-8") as f:
                json.dump({"paused": True}, f)
            return {"ok": True, "message": "Pause requested (flag written). The "
                    "orchestrator will honor it at the next stage boundary."}
        else:
            if os.path.exists(_CONTROL_FILE):
                os.remove(_CONTROL_FILE)
            return {"ok": True, "message": "Resume requested (flag cleared)."}
    except OSError as exc:
        return {"ok": False, "message": f"Could not update control flag: {exc}"}


def is_paused():
    try:
        if not os.path.exists(_CONTROL_FILE):
            return False
        with open(_CONTROL_FILE, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("paused"))
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #

class DashboardHandler(BaseHTTPRequestHandler):
    # Populated in main().
    log_path = "full_run.log"

    # Silence default per-request stderr logging (keeps the console clean).
    def log_message(self, fmt, *args):  # noqa: A003
        return

    # ---- helpers ---- #
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- routing ---- #
    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send_file(_INDEX_HTML, "text/html; charset=utf-8")
            return
        if self.path.startswith("/api/status"):
            self._handle_status()
            return
        self.send_error(404, "Not found")

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/control"):
            self._handle_control()
            return
        self.send_error(404, "Not found")

    # ---- endpoints ---- #
    def _read_log(self):
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None

    def _handle_status(self):
        raw = self._read_log()
        log_exists = raw is not None
        status = parse_status(raw or "")

        # Overlay "stopped": no [done] line AND the run process is not alive.
        running = bool(find_run_pids())
        if status["phase"] != "done" and not running and log_exists and raw.strip():
            # Only call it "stopped" once there *was* real output to begin with.
            status["phase"] = "stopped"

        try:
            mtime = os.path.getmtime(self.log_path) if log_exists else None
        except OSError:
            mtime = None

        # Creep the bar through the log-silent v2 conformal phase (anchored to the
        # log's last-write time = when conformal began). No effect outside it.
        apply_v2_conformal_creep(status, time.time(), mtime)

        payload = {
            "ok": True,
            "log_exists": log_exists,
            "log_path": os.path.abspath(self.log_path),
            "log_mtime": mtime,
            "server_time": time.time(),
            "run_detected": running,
            "paused": is_paused(),
            "gpu": get_gpu(),
            "sys": get_sys(),
            "status": status,
        }
        self._send_json(payload)

    def _handle_control(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            self._send_json({"ok": False, "message": "Invalid JSON body."}, code=400)
            return
        action = (body.get("action") or "").lower()
        if action == "stop":
            self._send_json(stop_run())
        elif action == "pause":
            self._send_json(set_paused(True))
        elif action == "resume":
            self._send_json(set_paused(False))
        else:
            self._send_json(
                {"ok": False, "message": f"Unknown action: {action!r}"}, code=400
            )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="NeuroForge live research dashboard.")
    p.add_argument("--log", default="full_run.log", help="path to the run log file")
    p.add_argument("--port", type=int, default=8765, help="HTTP port")
    p.add_argument("--host", default="127.0.0.1", help="bind address")
    p.add_argument("--open", action="store_true", help="open a browser on start")
    p.add_argument("--run-match", default="run_full_research",
                   help="substring identifying the run process for Stop/alive "
                        "detection; use 'run_certificates' for a single/cert run")
    p.add_argument("--bb-epochs", type=int, default=40,
                   help="single-run backbone epoch budget (bar scaling)")
    p.add_argument("--corr-epochs", type=int, default=15,
                   help="single-run corrector epoch budget (bar scaling)")
    args = p.parse_args(argv)

    global RUN_MATCH, SINGLE_BB_EPOCHS, SINGLE_CORR_EPOCHS
    RUN_MATCH = args.run_match
    SINGLE_BB_EPOCHS = args.bb_epochs
    SINGLE_CORR_EPOCHS = args.corr_epochs

    # Resolve the log relative to CWD; fall back to repo root if not found there
    # (so launching from anywhere still finds the canonical full_run.log).
    log_path = args.log
    if not os.path.isabs(log_path) and not os.path.exists(log_path):
        alt = os.path.join(_REPO_ROOT, args.log)
        if os.path.exists(alt):
            log_path = alt
    DashboardHandler.log_path = os.path.abspath(log_path)

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"NeuroForge dashboard -> {url}")
    print(f"  watching log : {DashboardHandler.log_path}")
    print("  (read-only; zero coupling to the training process)  Ctrl-C to stop.")

    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down dashboard.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
