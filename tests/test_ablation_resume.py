"""Per-seed checkpoint + resume in the ablation harness (Change C).

Fast + synthetic (res 24, n_train 4, n_val 2, 1 epoch, seeds (0, 1)). We run
``run_ablation`` once, confirm per-seed checkpoints were written, delete the
final CSV but keep the checkpoints, then re-run and confirm it resumed (skipped)
the seeds and reproduced a byte-identical final table.
"""

from __future__ import annotations

import os

import pytest

from benchmarks.ablation import run_ablation, run_ood_ablation, _seed_ckpt_path


@pytest.mark.slow
def test_ablation_resume_reproduces_table(tmp_path, capsys):
    out = str(tmp_path)
    kw = dict(
        n_train=4, n_val=2, resolution=24, seeds=(0, 1),
        epochs=1, corrector_epochs=1, width=10, modes=4, n_layers=2,
        batch_size=2, download=False, device="cpu", out_dir=out, verbose=True,
    )

    # First run: trains everything; checkpoints each seed.
    run_ablation("synthetic", **kw)
    for seed in (0, 1):
        assert os.path.exists(_seed_ckpt_path(out, seed)), f"missing checkpoint seed{seed}"
    csv_path = os.path.join(out, "ablation.csv")
    md_path = os.path.join(out, "ablation.md")
    first_csv = open(csv_path, "rb").read()
    first_md = open(md_path, "rb").read()

    # Delete the final outputs but KEEP the seed checkpoints.
    os.remove(csv_path)
    os.remove(md_path)

    # Second run: must skip both seeds (resume) and reproduce identical tables.
    capsys.readouterr()  # clear captured output
    run_ablation("synthetic", **kw)
    printed = capsys.readouterr().out
    assert printed.count("resumed from checkpoint (skip)") == 2

    assert open(csv_path, "rb").read() == first_csv, "CSV differs after resume"
    assert open(md_path, "rb").read() == first_md, "MD differs after resume"


@pytest.mark.slow
def test_ood_ablation_resume_reproduces_table(tmp_path, capsys):
    """OOD resume keys by (task, seed); per-seed file holds {task: {arm: ...}}."""
    out = str(tmp_path)
    kw = dict(
        ood_tasks=("scarce", "full"), n_train=4, n_val=2, resolution=24, seeds=(0,),
        epochs=1, corrector_epochs=1, width=10, modes=4, n_layers=2,
        batch_size=2, download=False, device="cpu", out_dir=out, verbose=True,
    )

    run_ood_ablation("synthetic", **kw)
    assert os.path.exists(_seed_ckpt_path(out, 0))
    csv_path = os.path.join(out, "ablation_ood.csv")
    md_path = os.path.join(out, "ablation_ood.md")
    first_csv = open(csv_path, "rb").read()
    first_md = open(md_path, "rb").read()

    os.remove(csv_path)
    os.remove(md_path)

    capsys.readouterr()
    run_ood_ablation("synthetic", **kw)
    printed = capsys.readouterr().out
    # One (task, seed) skip line per OOD task.
    assert printed.count("resumed from checkpoint (skip)") == 2

    assert open(csv_path, "rb").read() == first_csv, "OOD CSV differs after resume"
    assert open(md_path, "rb").read() == first_md, "OOD MD differs after resume"
