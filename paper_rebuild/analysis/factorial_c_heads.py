"""Factorial cell (c): direct scalar heads on frozen Transolver features (readout vs representation).

PREREG_factorial.md cell (c), unchanged, plus one secondary logged in DEVIATIONS.md
before this script first ran:
  embedding (primary)   : mean over ALL nodes of the last hidden layer
                          (out_norm output, width 256), per case, per seed.
  embedding (secondary) : mean over SURFACE nodes only (forces are surface integrals).
  heads                 : (c1) embedding only; (c2) embedding + the 6 raw case parameters
                          [U, alpha, naca_0..3]. Families Q and K exactly as
                          null_reference.py (training-CV chooses family and hyperparameters).
  data                  : official `full` split, train 800 -> heads fit; test 200 -> scored.
  seeds                 : Transolver seeds 0-4; per-case error averaged over seeds (primary),
                          log(1 - rho) averaged over seeds.
Outcomes and pairing as factorial_b_compare.py: per-case |error| in drag counts (C_D) and
absolute (C_L); log(1 - rho); paired differences vs the reference R, B = 10000.
Embeddings are cached in data/cache/factorial_c_emb.npz (untracked, regenerable).
Labels: official Simulation.force_coefficient(reference=True), read from the nullbench CSV
(case_id, cd, cl), which was built from those labels.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import neuroforge  # noqa: F401
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from null_reference import DATA, ROOT, predict_both  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
CACHE = os.path.join(ROOT, "data", "cache", "factorial_c_emb.npz")
OUT = os.path.join(ROOT, "paper_rebuild", "results", "factorial_c_heads.json")
SEEDS = [0, 1, 2, 3, 4]
PCOLS = ["U", "alpha", "naca_0", "naca_1", "naca_2", "naca_3"]


def embed_all():
    import torch
    from neuroforge.data.pointcloud import load_airfrans_pointclouds
    from recompute_force_vs_official import load_backbone

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for split, train in (("train", True), ("test", False)):
        pcs = load_airfrans_pointclouds(root="data", task="full", train=train, cache_dir="data/cache")
        out[f"{split}_names"] = np.array([pc.name for pc in pcs])
        for s in SEEDS:
            m, pn, _g, _n = load_backbone(os.path.join(ROOT, "checkpoints", "v2_transolver", f"seed{s}.pt"), dev)
            feats_all, feats_surf = [], []
            for pc in pcs:
                x = torch.from_numpy(pn.transform_in(pc.features)).float().to(dev).unsqueeze(0)
                with torch.no_grad():
                    fx = m.embed(x)
                    for blk in m.blocks:
                        fx = blk(fx)
                    h = m.out_norm(fx).squeeze(0)
                surf = torch.from_numpy(pc.targets[:, 0] == 0).to(dev)  # AirfRANS surface flag: u == 0
                feats_all.append(h.mean(0).cpu().numpy())
                feats_surf.append(h[surf].mean(0).cpu().numpy())
            out[f"{split}_all_s{s}"] = np.array(feats_all)
            out[f"{split}_surf_s{s}"] = np.array(feats_surf)
            print(split, "seed", s, "done", flush=True)
        del pcs
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **out)


def l1r(y, p):
    return float(np.log(max(1.0 - spearmanr(y, p).correlation, 1e-12)))


def boot(fn, n, b=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(b)])
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def main():
    if not os.path.exists(CACHE):
        embed_all()
    E = np.load(CACHE)
    with open(os.path.join(DATA, "airfrans", "airfrans_full_800_200.csv"), newline="") as fh:
        rows = {r["case_id"]: r for r in csv.DictReader(fh)}
    trn, ten = list(E["train_names"]), list(E["test_names"])
    Ptr = np.array([[float(rows[n][c]) for c in PCOLS] for n in trn])
    Pte = np.array([[float(rows[n][c]) for c in PCOLS] for n in ten])
    out = {"n_train": len(trn), "n_test": len(ten)}
    for t, scale in (("cd", 1e4), ("cl", 1.0)):
        ytr = np.array([float(rows[n][t]) for n in trn])
        yte = np.array([float(rows[n][t]) for n in ten])
        pq, pk, h = predict_both(Ptr, ytr, Pte)
        ref = pk if h["ref"] == "K" else pq
        err_ref = np.abs(ref - yte) * scale
        res = {"reference": {"mae": float(err_ref.mean()), "log1m_rho": l1r(yte, ref), "family": h["ref"]}}
        for pool in ("all", "surf"):
            for head in ("c1", "c2"):
                preds, fams = [], []
                for s in SEEDS:
                    Xtr, Xte = E[f"train_{pool}_s{s}"], E[f"test_{pool}_s{s}"]
                    if head == "c2":
                        Xtr, Xte = np.hstack([Xtr, Ptr]), np.hstack([Xte, Pte])
                    a, b_, hh = predict_both(Xtr, ytr, Xte)
                    preds.append(b_ if hh["ref"] == "K" else a)
                    fams.append(hh["ref"])
                P = np.array(preds).T
                err = np.abs(P - yte[:, None]).mean(1) * scale
                d = err - err_ref
                l1 = [l1r(yte, P[:, k]) for k in range(len(SEEDS))]
                res[f"{head}_{pool}"] = {
                    "families": fams, "mae_seedavg": float(err.mean()),
                    "mae_per_seed": [float(np.abs(P[:, k] - yte).mean() * scale) for k in range(len(SEEDS))],
                    "log1m_rho_seedavg": float(np.mean(l1)), "log1m_rho_per_seed": l1,
                    "delta_mae_vs_ref": {"point": float(d.mean()), "pct95": boot(lambda i: d[i].mean(), len(d)),
                                         "cases_ref_better": int((d > 0).sum())},
                    "delta_log1m_rho_vs_ref": float(np.mean(l1)) - res["reference"]["log1m_rho"]}
                print(t, head, pool, round(float(err.mean()), 4), "ref", round(float(err_ref.mean()), 4), flush=True)
        out[t] = res
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
