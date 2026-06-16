"""Reproducible, multi-seed ablation harness — reviewer-grade evidence.

Answers the questions a referee will ask, with statistics:

* **Does the corrector improve accuracy** (field MSE, rho_Cd) — not just the
  residual? (backbone vs backbone+corrector, same backbone.)
* **DEQ vs feed-forward corrector** — does the principled fixed point win?
* **Does the physics loss help or hurt** the end metrics?
* **Is the residual a valid trust signal** (residual<->error correlation)?

Each arm is trained over several seeds; results are reported as mean ± std and
written to a markdown table + CSV. Smoke-test on the bundled synthetic substrate
(CPU); run for real with ``source='airfrans'`` on a GPU.

    from benchmarks.ablation import run_ablation
    run_ablation('airfrans', task='full', n_train=400, n_val=120,
                 cache_dir='data/cache', seeds=(0, 1, 2))
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

__all__ = ["run_ablation", "run_ood_ablation"]

# Sidecar directory (under out_dir) for per-seed incremental checkpoints, so a
# crash mid-stage doesn't lose completed seeds' training.
_CKPT_SUBDIR = "_seed_checkpoints"


def _seed_ckpt_path(out_dir: str, seed) -> str:
    return os.path.join(out_dir, _CKPT_SUBDIR, f"seed{seed}.json")


def _save_seed_checkpoint(out_dir: str, seed, payload: dict) -> None:
    """Atomically persist one seed's per-arm metric dicts (temp-file + rename).

    Written once, after all of the seed's arms complete, so resume never treats a
    partially-written seed as done.
    """
    path = _seed_ckpt_path(out_dir, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _load_seed_checkpoint(out_dir: str, seed):
    """Return a seed's checkpoint payload, or ``None`` if absent/corrupt."""
    path = _seed_ckpt_path(out_dir, seed)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # corrupt/partial file -> recompute this seed
        return None

# AirfRANS-protocol metrics reported per arm.
_METRICS = [
    "mse_u", "mse_v", "mse_p", "surface_mse_p",
    "rho_cl", "rho_cd", "residual_error_spearman",
]


def _agg(dicts: list[dict]) -> dict[str, tuple[float, float]]:
    """Mean ± std (over seeds) for each metric, ignoring NaNs."""
    out = {}
    for m in _METRICS:
        vals = [d[m] for d in dicts if m in d and np.isfinite(d[m])]
        out[m] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
    return out


def _to_markdown(table: dict, seeds, source: str, task: str) -> str:
    hdr = "| arm | " + " | ".join(_METRICS) + " |"
    sep = "|" + "---|" * (len(_METRICS) + 1)
    lines = [
        f"### Ablation — {source}/{task}, {len(list(seeds))} seeds (mean ± std)",
        "", hdr, sep,
    ]
    for arm, mets in table.items():
        cells = []
        for m in _METRICS:
            mu, sd = mets[m]
            cells.append("n/a" if np.isnan(mu) else f"{mu:.3f}±{sd:.3f}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "_Lower MSE is better; rho closer to 1 is better; "
        "`residual_error_spearman` > 0 means a low residual tracks low error "
        "(the trust signal is valid). The corrector helps iff "
        "`backbone + corrector` beats `backbone` on MSE / rho._",
    ]
    return "\n".join(lines)


def _to_csv(table: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "metric", "mean", "std"])
        for arm, mets in table.items():
            for m, (mu, sd) in mets.items():
                w.writerow([arm, m, mu, sd])


def run_ablation(
    source: str = "synthetic",
    *,
    task: str = "scarce",
    n_train: int = 200,
    n_val: int = 100,
    resolution: int = 128,
    seeds=(0, 1, 2),
    epochs: int = 80,
    corrector_epochs: int = 20,
    width: int = 48,
    modes: int = 20,
    n_layers: int = 4,
    batch_size: int = 8,
    root: str = "data",
    cache_dir: str | None = None,
    download: bool = True,
    device: str = "auto",
    out_dir: str = "results",
    verbose: bool = True,
    resume: bool = True,
) -> dict:
    """Train every ablation arm over ``seeds`` and emit a mean±std table.

    Arms: ``backbone``, ``backbone (no physics loss)``,
    ``backbone + local corrector``, ``backbone + DEQ corrector``. The DEQ arm's
    backbone is reused for the plain ``backbone`` row (same trained model,
    evaluated with the correction off vs on) so the corrector comparison is
    apples-to-apples.

    When ``resume`` is True (default) each seed's per-arm metric dicts are
    checkpointed to ``out_dir/_seed_checkpoints/seed{N}.json`` after the seed's
    arms finish; on a later call those seeds are loaded and skipped. The final
    table aggregates checkpoints + freshly-run seeds, so it is identical to a
    non-resumed run on the same seeds (``_agg`` is order-insensitive).
    """
    import neuroforge as nf

    def _run_seed(seed) -> dict[str, dict]:
        """Train + evaluate all arms for one seed -> {arm: metric dict}."""
        common = dict(
            backbone="fno", width=width, modes=modes, n_layers=n_layers,
            dropout=0.05, epochs=epochs, corrector_epochs=corrector_epochs,
            resolution=resolution, batch_size=batch_size, device=device, seed=seed,
        )
        fit_kw = dict(
            task=task, n_train=n_train, n_val=n_val, root=root,
            cache_dir=cache_dir, download=download, verbose=False,
        )
        seed_arms: dict[str, dict] = {}

        if verbose:
            print(f"[ablation] seed {seed}: DEQ-corrector arm ...")
        m = nf.NeuroForge(corrector="deq", physics_weight=0.1, **common).fit(source, **fit_kw)
        seed_arms["backbone"] = m.evaluate(corrected=False)
        seed_arms["backbone + DEQ corrector"] = m.evaluate(corrected=True)

        if verbose:
            print(f"[ablation] seed {seed}: local-corrector arm ...")
        ml = nf.NeuroForge(corrector="local", physics_weight=0.1, **common).fit(source, **fit_kw)
        seed_arms["backbone + local corrector"] = ml.evaluate(corrected=True)

        if verbose:
            print(f"[ablation] seed {seed}: no-physics-loss arm ...")
        mn = nf.NeuroForge(corrector="none", physics_weight=0.0, **common).fit(source, **fit_kw)
        seed_arms["backbone (no physics loss)"] = mn.evaluate(corrected=False)
        return seed_arms

    arms: dict[str, list[dict]] = defaultdict(list)
    for seed in seeds:
        seed_arms = _load_seed_checkpoint(out_dir, seed) if resume else None
        if seed_arms is not None:
            if verbose:
                print(f"[ablation] seed {seed}: resumed from checkpoint (skip)")
        else:
            seed_arms = _run_seed(seed)
            if resume:
                _save_seed_checkpoint(out_dir, seed, seed_arms)
        for arm, mets in seed_arms.items():
            arms[arm].append(mets)

    order = [
        "backbone",
        "backbone (no physics loss)",
        "backbone + local corrector",
        "backbone + DEQ corrector",
    ]
    table = {arm: _agg(arms[arm]) for arm in order if arm in arms}

    md = _to_markdown(table, seeds, source, task)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ablation.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    _to_csv(table, os.path.join(out_dir, "ablation.csv"))
    if verbose:
        print("\n" + md)
        print(f"\n[ablation] wrote {out_dir}/ablation.md and ablation.csv")
    return {"table": table, "seeds": list(seeds), "source": source, "task": task}


def _ood_to_markdown(tables: dict, seeds, source: str, ood_tasks) -> str:
    """One mean±std section per OOD task, with a prominent OOD note.

    ``tables`` maps ``task -> {arm: {metric: (mean, std)}}``.
    """
    n_seeds = len(list(seeds))
    lines = [
        f"### OOD ablation — {source}, {n_seeds} seeds (mean ± std)",
        "",
        "**OUT-OF-DISTRIBUTION evaluation.** Each arm is trained on a task's "
        "*train* split and evaluated on that task's *test* split, whose "
        f"{source}-protocol parameter range (e.g. Reynolds number / angle of "
        "attack) is **disjoint** from the train range. These metrics therefore "
        "measure generalization to unseen flow regimes, not in-distribution fit.",
        "",
    ]
    for task in ood_tasks:
        table = tables[task]
        hdr = "| arm | " + " | ".join(_METRICS) + " |"
        sep = "|" + "---|" * (len(_METRICS) + 1)
        lines += [f"#### OOD task: `{task}` (train range -> held-out test range)", "", hdr, sep]
        for arm, mets in table.items():
            cells = []
            for m in _METRICS:
                mu, sd = mets[m]
                cells.append("n/a" if np.isnan(mu) else f"{mu:.3f}±{sd:.3f}")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "_Lower MSE is better; rho closer to 1 is better. The correction loop "
        "reduces the in-distribution -> OOD gap iff `backbone + corrector` beats "
        "`backbone` on these (OOD) metrics by a wider margin than it does "
        "in-distribution. Compare against the in-distribution `run_ablation` "
        "table on the same arms to read off the gap._",
    ]
    return "\n".join(lines)


def _ood_to_csv(tables: dict, path: str) -> None:
    """Flat CSV with a leading ``ood_task`` column across all OOD tasks."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ood_task", "arm", "metric", "mean", "std"])
        for task, table in tables.items():
            for arm, mets in table.items():
                for m, (mu, sd) in mets.items():
                    w.writerow([task, arm, m, mu, sd])


def run_ood_ablation(
    source: str = "airfrans",
    *,
    ood_tasks=("reynolds", "aoa"),
    n_train: int = 400,
    n_val: int = 120,
    resolution: int = 128,
    seeds=(0, 1, 2),
    epochs: int = 80,
    corrector_epochs: int = 20,
    width: int = 48,
    modes: int = 20,
    n_layers: int = 4,
    batch_size: int = 8,
    root: str = "data",
    cache_dir: str | None = None,
    download: bool = True,
    device: str = "auto",
    out_dir: str = "results",
    verbose: bool = True,
    resume: bool = True,
) -> dict:
    """Out-of-distribution ablation: per-arm metrics on each task's held-out range.

    For every task in ``ood_tasks`` (e.g. AirfRANS ``'reynolds'`` / ``'aoa'``,
    whose train/test splits cover *disjoint* parameter ranges), every ablation
    arm is trained on the task's train split and evaluated on its test split via
    :meth:`NeuroForge.evaluate` (which defaults to the held-out validation/test
    pairs). Training on the train range and scoring on the disjoint test range
    *is* the OOD test the reviewers asked for; the resulting table makes the
    in-distribution → OOD gap legible and shows whether the correction loop
    reduces it.

    Arms mirror :func:`run_ablation` exactly: ``backbone``,
    ``backbone (no physics loss)``, ``backbone + local corrector``,
    ``backbone + DEQ corrector``. The DEQ arm's backbone is reused for the plain
    ``backbone`` row (correction off vs on) so the comparison is apples-to-apples.
    Results are aggregated to mean ± std over ``seeds`` and written to
    ``ablation_ood.md`` / ``ablation_ood.csv``.

    When ``resume`` is True (default) each seed is checkpointed to
    ``out_dir/_seed_checkpoints/seed{N}.json`` keyed by task
    (``{task: {arm: metrics}}``) once all of that (task, seed)'s arms finish, and
    a later call loads + skips completed (task, seed) pairs. The final table is
    identical to a non-resumed run on the same seeds.
    """
    import neuroforge as nf

    tables: dict[str, dict] = {}
    raw: dict[str, dict[str, list[dict]]] = {}
    order = [
        "backbone",
        "backbone (no physics loss)",
        "backbone + local corrector",
        "backbone + DEQ corrector",
    ]

    def _run_task_seed(task, seed) -> dict[str, dict]:
        """Train + evaluate all arms for one (task, seed) -> {arm: metric dict}."""
        common = dict(
            backbone="fno", width=width, modes=modes, n_layers=n_layers,
            dropout=0.05, epochs=epochs, corrector_epochs=corrector_epochs,
            resolution=resolution, batch_size=batch_size, device=device, seed=seed,
        )
        fit_kw = dict(
            task=task, n_train=n_train, n_val=n_val, root=root,
            cache_dir=cache_dir, download=download, verbose=False,
        )
        seed_arms: dict[str, dict] = {}

        if verbose:
            print(f"[ood-ablation] {task} | seed {seed}: DEQ-corrector arm ...")
        m = nf.NeuroForge(corrector="deq", physics_weight=0.1, **common).fit(source, **fit_kw)
        seed_arms["backbone"] = m.evaluate(corrected=False)
        seed_arms["backbone + DEQ corrector"] = m.evaluate(corrected=True)

        if verbose:
            print(f"[ood-ablation] {task} | seed {seed}: local-corrector arm ...")
        ml = nf.NeuroForge(corrector="local", physics_weight=0.1, **common).fit(source, **fit_kw)
        seed_arms["backbone + local corrector"] = ml.evaluate(corrected=True)

        if verbose:
            print(f"[ood-ablation] {task} | seed {seed}: no-physics-loss arm ...")
        mn = nf.NeuroForge(corrector="none", physics_weight=0.0, **common).fit(source, **fit_kw)
        seed_arms["backbone (no physics loss)"] = mn.evaluate(corrected=False)
        return seed_arms

    for task in ood_tasks:
        if verbose:
            print(f"[ood-ablation] OOD task '{task}': train split -> held-out test split")
        arms: dict[str, list[dict]] = defaultdict(list)
        for seed in seeds:
            ck = _load_seed_checkpoint(out_dir, seed) if resume else None
            seed_arms = ck.get(task) if isinstance(ck, dict) else None
            if seed_arms is not None:
                if verbose:
                    print(f"[ood-ablation] {task} | seed {seed}: resumed from checkpoint (skip)")
            else:
                seed_arms = _run_task_seed(task, seed)
                if resume:
                    # Merge into this seed's (multi-task) checkpoint atomically.
                    payload = ck if isinstance(ck, dict) else {}
                    payload[task] = seed_arms
                    _save_seed_checkpoint(out_dir, seed, payload)
            for arm, mets in seed_arms.items():
                arms[arm].append(mets)

        raw[task] = dict(arms)
        tables[task] = {arm: _agg(arms[arm]) for arm in order if arm in arms}

    md = _ood_to_markdown(tables, seeds, source, ood_tasks)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ablation_ood.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    _ood_to_csv(tables, os.path.join(out_dir, "ablation_ood.csv"))
    if verbose:
        print("\n" + md)
        print(f"\n[ood-ablation] wrote {out_dir}/ablation_ood.md and ablation_ood.csv")
    return {
        "tables": tables, "seeds": list(seeds), "source": source,
        "ood_tasks": list(ood_tasks),
    }


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="NeuroForge reproducible ablation.")
    p.add_argument("--source", default="airfrans", choices=["airfrans", "synthetic"])
    p.add_argument("--task", default="full")
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-val", type=int, default=120)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--corrector-epochs", type=int, default=20)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="results")
    p.add_argument(
        "--ood", action="store_true",
        help="Run the out-of-distribution ablation (per OOD task, train split -> "
             "held-out test range) instead of the in-distribution ablation.",
    )
    p.add_argument(
        "--ood-tasks", nargs="+", default=["reynolds", "aoa"],
        help="OOD tasks to evaluate (each has disjoint train/test parameter ranges).",
    )
    a = p.parse_args(argv)
    if a.ood:
        run_ood_ablation(
            a.source, ood_tasks=tuple(a.ood_tasks), n_train=a.n_train, n_val=a.n_val,
            resolution=a.res, seeds=tuple(a.seeds), epochs=a.epochs,
            corrector_epochs=a.corrector_epochs, cache_dir=a.cache_dir, out_dir=a.out_dir,
        )
        return 0
    run_ablation(
        a.source, task=a.task, n_train=a.n_train, n_val=a.n_val, resolution=a.res,
        seeds=tuple(a.seeds), epochs=a.epochs, corrector_epochs=a.corrector_epochs,
        cache_dir=a.cache_dir, out_dir=a.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
