"""Fast tests for the decoupled dashboard log parser (`scripts/dashboard/server.py`).

The parser is a pure function over the raw research log. These tests feed it a
synthetic multi-line log — including tqdm carriage-return (\\r) segments — and a
deliberately *short* log, asserting it extracts stage / seed / arm / epoch /
latest losses / a 0..100 progress value and degrades gracefully.

No neuroforge / torch / numpy import; nothing here touches training.
"""

import os
import sys

# scripts/dashboard is not a package on sys.path (there is no scripts/__init__.py
# by design), so import server.py directly by adding its directory.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DASH = os.path.join(_REPO, "scripts", "dashboard")
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

import server  # noqa: E402
from server import parse_status  # noqa: E402


# A realistic log: header, stage 1, an ablation arm, a long tqdm rasterise bar
# (joined with \r as tqdm does), then real epoch lines, a step line and a
# corrector-epoch line. Mirrors the formats in trainer.py / ablation.py.
def _full_log():
    rasterise = "\r".join(
        f"rasterise airfrans/full/train:  {int(100*i/8)}%|##| {i}/800 "
        f"[00:{i:02d}<10:00,  6.0s/it]"
        for i in range(1, 6)
    )
    loading = "\r".join(
        f"Loading dataset (task: full, split: train):  {int(100*i/8)}%|##| "
        f"{i}/800 [00:0{i}<02:00,  5.5it/s]"
        for i in range(1, 4)
    )
    return "\n".join([
        "================================================================",
        "NeuroForge — full research run   (preset: full)",
        "  data     : 800 train / 200 val   seeds=[0, 1, 2]",
        "",
        "[stage 1] in-distribution ablation ...",
        "[ablation] seed 0: DEQ-corrector arm ...",
        loading,
        rasterise,
        "[train] AMP enabled (dtype=bfloat16)",
        "[epoch 0] train_loss=1.2345e-01 val_loss=2.3456e-01",
        "[train] epoch 1 step 50 loss=9.0e-02 data=8e-02 phys=1e-02 bc=0e+00",
        "[epoch 1] train_loss=8.0000e-02 val_loss=1.5000e-01",
        "[epoch 2] train_loss=5.5000e-02 val_loss=1.1000e-01",
        "[corrector epoch 0] loss=3.3000e-02 per_channel=[1e-2, 1e-2, 1e-2, 0e0]",
    ])


def test_extracts_stage_seed_arm_epoch_and_losses():
    s = parse_status(_full_log())

    assert s["stage"] == "in_dist"
    assert s["seed"] == 0
    assert s["arm"].lower() == "deq-corrector"
    # epochs are 0-indexed in the log; latest is epoch 2
    assert s["epoch"] == 2
    assert abs(s["train_loss"] - 5.5e-2) < 1e-9
    assert abs(s["val_loss"] - 1.1e-1) < 1e-9
    # corrector epoch parsed too
    assert s["corrector_epoch"] == 0
    # loss history collects all epoch train_losses in order
    assert s["loss_history"] == [1.2345e-01, 8.0e-02, 5.5e-02]
    # phase is training once epochs appear
    assert s["phase"] == "training"
    # progress is a sane in-range float and non-trivial
    assert isinstance(s["progress"], float)
    assert 0.0 <= s["progress"] <= 100.0
    assert s["progress"] > 0.0


def test_tqdm_latest_segment_wins_for_dataprep():
    # A log that stops *during* rasterise (no epochs yet) — common early state.
    log = "\n".join([
        "[stage 1] in-distribution ablation ...",
        "[ablation] seed 0: DEQ-corrector arm ...",
        "\r".join(
            f"rasterise airfrans/full/train:  30%|##| {c}/800 [22:00<60:00, 6s/it]"
            for c in (240, 241, 242)
        ),
    ])
    s = parse_status(log)
    assert s["phase"] == "preparing"
    assert s["dataprep"] is not None
    # latest \r segment wins -> 242, not 240
    assert s["dataprep"]["cur"] == 242
    assert s["dataprep"]["tot"] == 800
    assert s["epoch"] is None
    assert s["train_loss"] is None
    # progress moves during data-prep (must be > 0 but small, well under 100)
    assert 0.0 < s["progress"] < 50.0


def test_progress_climbs_with_rasterise_fraction():
    """The dedicated front slice must advance as the initial rasterise advances
    (so the bar is not visually pinned at ~0 during the long data load)."""
    def at(cur):
        log = "\n".join([
            "[stage 1] in-distribution ablation ...",
            "[ablation] seed 0: DEQ-corrector arm ...",
            f"rasterise airfrans/full/train:  50%|##| {cur}/800 [10:00<10:00, 6s/it]",
        ])
        return parse_status(log)["progress"]

    p10, p400, p800 = at(80), at(400), at(800)
    assert p10 < p400 < p800            # strictly climbing with the load
    assert p800 <= 5.0 + 1e-6           # but capped within the reserved front slice
    assert p10 > 0.0


def test_ood_stage_arm_and_task():
    log = "\n".join([
        "[stage 1] in-distribution ablation ...",
        "[ablation] seed 2: no-physics-loss arm ...",
        "[epoch 79] train_loss=1.0e-03 val_loss=2.0e-03",
        "[stage 2] out-of-distribution ablation ...",
        "[ood-ablation] reynolds | seed 0: DEQ-corrector arm ...",
        "[ood-ablation] OOD task 'reynolds': train split -> held-out test split",
        "[epoch 0] train_loss=5.0e-02 val_loss=6.0e-02",
    ])
    s = parse_status(log)
    assert s["stage"] == "ood"
    assert s["ood_task"] == "reynolds"
    assert s["seed"] == 0
    assert s["arm"].lower() == "deq-corrector"
    assert s["epoch"] == 0
    # OOD is the back two-thirds of the run, so progress should be well past 33%
    assert s["progress"] > 33.0


def test_done_line_pins_progress_at_100():
    log = "\n".join([
        "[stage 1] in-distribution ablation ...",
        "[stage 2] out-of-distribution ablation ...",
        "[done] all stages complete. See results/full_research/SUMMARY.md",
    ])
    s = parse_status(log)
    assert s["stage"] == "done"
    assert s["phase"] == "done"
    assert s["progress"] == 100.0


def test_empty_and_short_logs_are_graceful():
    for txt in ("", "   ", "some unrelated noise\nmore noise"):
        s = parse_status(txt)
        assert s["stage"] in (None, "done")  # no crash
        assert s["seed"] is None
        assert s["arm"] is None
        assert s["epoch"] is None
        assert s["train_loss"] is None
        assert s["val_loss"] is None
        assert s["loss_history"] == []
        assert s["phase"] == "waiting"
        assert s["progress"] == 0.0
        assert 0.0 <= s["progress"] <= 100.0


def test_baseline_run_mode():
    """The Transolver baseline (scripts/run_baselines.py) trains `seeds` backbones
    of `bb` epochs each with NO corrector. The parser must recognise it, expose
    stage=="baseline" with the right seed/epoch, drive a monotonically increasing
    0..100 bar across seeds, and hit 100% on the final table2 line."""
    # Match the run shape: 3 seeds x 80 backbone epochs (set on the module so the
    # progress math uses the same budget the CLI would pass via --bb-epochs).
    server.SINGLE_BB_EPOCHS = 80
    bb = 80

    def at(seed, epoch):
        lines = [
            "[baseline] device=cuda",
            "[baseline] seed 0: transolver params = 7,654,321",
            f"[baseline] seed {seed} epoch {epoch}: train_mse=3.9557e-02 (77.6s)",
        ]
        return parse_status("\n".join(lines))

    # mid-run snapshot
    s = at(1, 26)
    assert s["stage"] == "baseline"
    assert s["seed"] == 1
    assert s["epoch"] == 26
    assert s["phase"] == "training"
    assert abs(s["train_loss"] - 3.9557e-02) < 1e-9
    assert s["loss_history"][-1] == 3.9557e-02
    assert "Baseline" in s["stage_label"]
    assert 1.0 <= s["progress"] <= 99.5

    # progress increases as the run advances across epochs and seeds
    p_s0_e0 = at(0, 0)["progress"]
    p_s0_e79 = at(0, 79)["progress"]
    p_s1_e0 = at(1, 0)["progress"]
    p_s2_e79 = at(2, 79)["progress"]
    assert p_s0_e0 < p_s0_e79 < p_s1_e0 < p_s2_e79
    # final backbone epoch of the last seed is essentially the whole grid
    assert p_s2_e79 <= 99.5

    # the table2 line pins the bar at 100%
    done = parse_status(
        "\n".join([
            "[baseline] seed 2 epoch 79: train_mse=1.0e-02 (77.6s)",
            "[baseline] wrote results/baselines/table2.md and table2.csv",
        ])
    )
    assert done["stage"] == "done"
    assert done["phase"] == "done"
    assert done["progress"] == 100.0


def test_v2_run_backbone_corrector_and_headline():
    """The v2 run (scripts/run_v2.py) trains a Transolver backbone, then runs the
    certified trust layer on its predictions. The parser must recognise stage
    'v2', track the per-seed sub-phase, capture the backbone wall-time for an ETA,
    reset epoch trackers at each new seed, and surface the headline residual<->error
    Spearman as it forms."""
    server.SINGLE_BB_EPOCHS = 80
    server.SINGLE_CORR_EPOCHS = 20

    # --- backbone epoch: stage/seed/epoch/loss/sec-per-epoch/ETA ---
    s = parse_status("\n".join([
        "[v2] device=cuda",
        "[v2] seed 0: Transolver params = 7,350,420",
        "[v2] seed 0 backbone epoch 11: train_mse=3.9557e-02 (45.0s)",
    ]))
    assert s["stage"] == "v2"
    assert s["seed"] == 0
    assert s["epoch"] == 11
    assert s["v2_phase"] == "backbone"
    assert abs(s["train_loss"] - 3.9557e-02) < 1e-9
    assert abs(s["sec_per_epoch"] - 45.0) < 1e-9
    assert s["phase"] == "training"
    # ETA: backbone epochs left this seed (80-12=68) + 2 future seeds * 80, plus a
    # tail allowance — must be a positive finite number, far more than one epoch.
    assert s["eta_seconds"] is not None and s["eta_seconds"] > 68 * 45.0

    # --- new seed resets the lingering corrector epoch from the previous seed ---
    s = parse_status("\n".join([
        "[v2] seed 0 corrector epoch 19: loss=1.0e-02",
        "[v2] ================= seed 1 =================",
        "[v2] seed 1: Transolver params = 7,350,420",
        "[v2] seed 1 backbone epoch 0: train_mse=2.0e-01 (44.0s)",
    ]))
    assert s["seed"] == 1
    assert s["epoch"] == 0
    assert s["corrector_epoch"] is None     # reset — did not leak from seed 0
    assert s["v2_phase"] == "backbone"

    # --- corrector phase + headline Spearman captured from the (a) eval line ---
    s = parse_status("\n".join([
        "[v2] seed 0: Transolver params = 7,350,420",
        "[v2] seed 0 backbone epoch 79: train_mse=1.0e-02 (45.0s)",
        "[v2] seed 0: evaluating (a) backbone alone ...",
        "[v2] seed 0: (a) eval 12.3s -> mse_u=0.12, rho_cl=0.99, "
        "residual_error_spearman=0.413",
        "[v2] seed 0: training deq corrector on Transolver residuals ...",
        "[v2] seed 0 corrector epoch 3: loss=2.2e-02",
    ]))
    assert s["v2_phase"] == "corrector"
    assert s["corrector_epoch"] == 3
    assert abs(s["v2_spearman"] - 0.413) < 1e-9   # the headline, on a SOTA backbone

    # --- done line pins 100% ---
    d = parse_status("\n".join([
        "[v2] seed 2 backbone epoch 79: train_mse=1.0e-02 (45.0s)",
        "[v2] ================= SUMMARY =================",
        "[v2] artifacts in results/v2",
    ]))
    assert d["stage"] == "v2"
    assert d["phase"] == "done"
    assert d["progress"] == 100.0


def test_v2_conformal_creep_advances_with_wallclock():
    """The conformal phase is log-silent for ~2.5h; apply_v2_conformal_creep must
    advance the bar from the wall-clock (anchored to log mtime) and be a no-op
    outside conformal — only ever raising progress (monotone)."""
    server.SINGLE_BB_EPOCHS = 80
    conf_log = "\n".join([
        "[v2] seed 0: Transolver params = 7,350,420",
        "[v2] seed 0 backbone epoch 79: train_mse=1e-2 (45s)",
        "[v2] seed 0: split-conformal coverage ...",
    ])
    s = parse_status(conf_log)
    assert s["v2_phase"] == "conformal"
    base = s["progress"]

    # elapsed 0 -> no advance
    s0 = parse_status(conf_log)
    server.apply_v2_conformal_creep(s0, now=1000.0, log_mtime=1000.0)
    assert abs(s0["progress"] - base) < 1e-6

    # halfway through the estimated duration -> clearly advanced
    s1 = parse_status(conf_log)
    server.apply_v2_conformal_creep(
        s1, now=1000.0 + server._V2_CONFORMAL_EST_SECS / 2, log_mtime=1000.0
    )
    assert s1["progress"] > base + 5.0
    assert s1["phase_elapsed_s"] > 0

    # outside conformal (backbone phase) -> no-op
    s2 = parse_status("[v2] seed 1 backbone epoch 10: train_mse=1e-1 (45s)")
    p2 = s2["progress"]
    server.apply_v2_conformal_creep(s2, now=99999.0, log_mtime=1000.0)
    assert abs(s2["progress"] - p2) < 1e-6


def test_v2_progress_is_monotone_across_seeds_and_phases():
    server.SINGLE_BB_EPOCHS = 80
    server.SINGLE_CORR_EPOCHS = 20
    snaps = [
        "[v2] seed 0: Transolver params = 7,350,420",
        "[v2] seed 0 backbone epoch 0: train_mse=2e-1 (45s)",
        "[v2] seed 0 backbone epoch 79: train_mse=1e-2 (45s)",
        "[v2] seed 0: evaluating (a) backbone alone ...",
        "[v2] seed 0: training deq corrector on Transolver residuals ...",
        "[v2] seed 0 corrector epoch 19: loss=1e-2",
        "[v2] seed 0: evaluating (b) backbone + deq loop ...",
        "[v2] seed 0: split-conformal coverage ...",
        "[v2] seed 1: Transolver params = 7,350,420",
        "[v2] seed 1 backbone epoch 40: train_mse=2e-2 (45s)",
        "[v2] seed 2 backbone epoch 79: train_mse=1e-2 (45s)",
        "[v2] artifacts in results/v2",
    ]
    last = -1.0
    ctx = ""
    for line in snaps:
        ctx += line + "\n"
        p = parse_status(ctx)["progress"]
        assert p >= last - 1e-6, f"v2 progress went backwards: {p} < {last} at {line!r}"
        last = p
    assert last == 100.0


def test_progress_is_monotone_as_run_advances():
    snapshots = [
        "[stage 1] in-distribution ablation ...",
        "[stage 1] in-distribution ablation ...\n[ablation] seed 0: DEQ-corrector arm ...",
        "[ablation] seed 0: DEQ-corrector arm ...\n[epoch 10] train_loss=1e-1 val_loss=1e-1",
        "[ablation] seed 1: DEQ-corrector arm ...\n[epoch 40] train_loss=1e-1 val_loss=1e-1",
        "[stage 2] out-of-distribution ablation ...\n"
        "[ood-ablation] aoa | seed 1: local-corrector arm ...\n"
        "[epoch 20] train_loss=1e-2 val_loss=1e-2",
        "[done] all stages complete.",
    ]
    last = -1.0
    for snap in snapshots:
        # seed each snapshot with the stage-1 marker so the parser knows context
        ctx = "[stage 1] in-distribution ablation ...\n" + snap
        p = parse_status(ctx)["progress"]
        assert p >= last - 1e-6, f"progress went backwards: {p} < {last}"
        last = p
    assert last == 100.0
