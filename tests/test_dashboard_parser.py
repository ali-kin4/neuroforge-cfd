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
