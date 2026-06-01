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
import os
from collections import defaultdict

import numpy as np

__all__ = ["run_ablation"]

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
) -> dict:
    """Train every ablation arm over ``seeds`` and emit a mean±std table.

    Arms: ``backbone``, ``backbone (no physics loss)``,
    ``backbone + local corrector``, ``backbone + DEQ corrector``. The DEQ arm's
    backbone is reused for the plain ``backbone`` row (same trained model,
    evaluated with the correction off vs on) so the corrector comparison is
    apples-to-apples.
    """
    import neuroforge as nf

    arms: dict[str, list[dict]] = defaultdict(list)
    for seed in seeds:
        common = dict(
            backbone="fno", width=width, modes=modes, n_layers=n_layers,
            dropout=0.05, epochs=epochs, corrector_epochs=corrector_epochs,
            resolution=resolution, batch_size=batch_size, device=device, seed=seed,
        )
        fit_kw = dict(
            task=task, n_train=n_train, n_val=n_val, root=root,
            cache_dir=cache_dir, download=download, verbose=False,
        )

        if verbose:
            print(f"[ablation] seed {seed}: DEQ-corrector arm ...")
        m = nf.NeuroForge(corrector="deq", physics_weight=0.1, **common).fit(source, **fit_kw)
        arms["backbone"].append(m.evaluate(corrected=False))
        arms["backbone + DEQ corrector"].append(m.evaluate(corrected=True))

        if verbose:
            print(f"[ablation] seed {seed}: local-corrector arm ...")
        ml = nf.NeuroForge(corrector="local", physics_weight=0.1, **common).fit(source, **fit_kw)
        arms["backbone + local corrector"].append(ml.evaluate(corrected=True))

        if verbose:
            print(f"[ablation] seed {seed}: no-physics-loss arm ...")
        mn = nf.NeuroForge(corrector="none", physics_weight=0.0, **common).fit(source, **fit_kw)
        arms["backbone (no physics loss)"].append(mn.evaluate(corrected=False))

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
    a = p.parse_args(argv)
    run_ablation(
        a.source, task=a.task, n_train=a.n_train, n_val=a.n_val, resolution=a.res,
        seeds=tuple(a.seeds), epochs=a.epochs, corrector_epochs=a.corrector_epochs,
        cache_dir=a.cache_dir, out_dir=a.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
