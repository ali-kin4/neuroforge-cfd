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

    state["progress"] = _compute_progress(
        state, saw_any_epoch_for_current_unit, last_dataprep
    )
    return state


# The one-time initial AirfRANS rasterise/load takes ~40 min (the dataset is
# cached afterwards). To avoid the bar sitting at ~0 for that whole phase, the
# first PREP_SLICE of the bar is dedicated to that initial data-prep, driven
# directly by the tqdm fraction; training then occupies the remaining
# (1 - PREP_SLICE). Smooth and strictly monotone across the handoff.
_PREP_SLICE = 0.05  # 5% of the overall bar reserved for the initial data load


def _compute_progress(state, saw_epoch_for_unit, dataprep) -> float:
    """Smooth, monotone 0..100 progress from the parsed state (see docstring)."""
    if state["phase"] == "done":
        return 100.0
    if state["stage"] is None:
        return 0.0

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
        "Where-Object { $_.CommandLine -match 'run_full_research' } | "
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
                if "run_full_research" not in line:
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
    runs_script = ("run_full_research.py" in low) or (
        "run_full_research" in low and ".py" in low
    )
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

        payload = {
            "ok": True,
            "log_exists": log_exists,
            "log_path": os.path.abspath(self.log_path),
            "log_mtime": mtime,
            "server_time": time.time(),
            "run_detected": running,
            "paused": is_paused(),
            "gpu": get_gpu(),
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
    args = p.parse_args(argv)

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
