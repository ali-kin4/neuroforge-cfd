"""Distil the five NullBench worked-example CSVs from committed sources.

Writes tidy ``(id, [split,] parameters..., label)`` CSVs under
``src/neuroforge/nullbench/data/<benchmark>/`` -- the exact contract
``neuroforge.nullbench.load_table`` reads, and small enough (parameters and
force labels only, no field data) to commit and ship with the package. Each
output file mirrors one nested feature set already reported and verified in
``docs/paper/review/null_travels.md`` (headline / richest-metadata row for
that benchmark), so ``neuroforge.nullbench.run_null`` on the shipped CSV
reproduces the committed number in ``results/review/covariate_null*.json``.

Sources (never re-distributed as-is; only the engineered columns actually
used are kept):

* AirfRANS -- ``results/control/_cache/official_labels_full_{test_n200,
  train_n800}.json`` (scalar case labels, already committed; the train file
  is generated once by this script's ``--airfrans-train-labels`` mode from
  the local AirfRANS install, which requires the (gitignored) raw dataset --
  not required to just rebuild the CSVs once that cache file exists).
* DrivAerML / DrivAerNet++ / AhmedML / WindsorML -- ``data/crossbench/*``,
  fetched by ``scripts/covariate_null_crossbench.py --download`` (gitignored,
  CC BY-NC / CC BY-SA metadata mirrors of the published Hugging Face /
  GitHub / Dropbox files listed in that script's ``SOURCES`` dict).

Usage
-----
    .venv/Scripts/python.exe scripts/nullbench_build_data.py

Pure stdlib + numpy, CPU-only, a few seconds.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy; see CLAUDE.md

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covariate_null import parse_case  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROSSBENCH = os.path.join(REPO, "data", "crossbench")
CACHE = os.path.join(REPO, "results", "control", "_cache")
OUT = os.path.join(REPO, "src", "neuroforge", "nullbench", "data")


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)))


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [{(k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
             for k, v in r.items()} for r in rows]


# --------------------------------------------------------------------------------------
# 1. AirfRANS -- official 800/200 split, richest metadata (U, alpha, alpha^2, NACA digits)
# --------------------------------------------------------------------------------------
def build_airfrans():
    test_path = os.path.join(CACHE, "official_labels_full_test_n200.json")
    train_path = os.path.join(CACHE, "official_labels_full_train_n800.json")
    with open(test_path, encoding="utf-8") as fh:
        test_labels = json.load(fh)
    with open(train_path, encoding="utf-8") as fh:
        train_labels = json.load(fh)

    # Train-row order never affects anything downstream (OLS is order-invariant, and
    # the reference bootstrap only resamples the TEST side), so it is sorted for a
    # readable diff. The TEST row order is NOT re-sorted: the committed cache already
    # preserves the AirfRANS dataset's own case order (verified: its keys are not
    # alphabetically sorted), and `scripts/covariate_null_trainfit.py`'s bootstrap
    # resamples the test array BY POSITION -- re-sorting here would still match the
    # point estimate (order-invariant) but silently change which cases land in each
    # bootstrap draw, and the CI would no longer reproduce bit-for-bit.
    names = sorted(train_labels) + list(test_labels)
    tags = ["train"] * len(train_labels) + ["test"] * len(test_labels)
    labels = {**train_labels, **test_labels}

    parsed = [parse_case(n) for n in names]
    width = max(len(d) for _, _, d in parsed)
    rows = []
    for nm, tag, (u, a, dig) in zip(names, tags, parsed):
        dig = dig + [0.0] * (width - len(dig))
        rows.append([nm, tag, u, a, a * a, *dig, labels[nm]["cl"], labels[nm]["cd"]])
    header = (["case_id", "split", "U", "alpha", "alpha2"]
              + ["naca_%d" % i for i in range(width)] + ["cl", "cd"])
    _write_csv(os.path.join(OUT, "airfrans", "airfrans_full_800_200.csv"), header, rows)


def build_airfrans_train_labels(root=os.path.join(REPO, "data", "Dataset")):
    """One-off: extract official (cl, cd) for the 800 `full`-train cases from
    the local AirfRANS install, mirroring `scripts/covariate_null_trainfit.py`
    exactly. Requires the (gitignored) raw dataset; writes only scalar labels,
    which is then a committed, non-field-data artifact like the existing
    200-case test cache."""
    import airfrans.dataset as afds

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from covariate_null_trainfit import official_labels

    _, train_names = afds.load(root=root, task="full", train=True)
    cl, cd, kept = official_labels(root, train_names)
    out = {nm: {"cl": float(c_l), "cd": float(c_d)} for nm, c_l, c_d in zip(kept, cl, cd)}
    dst = os.path.join(CACHE, "official_labels_full_train_n800.json")
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote %s (%d cases)" % (dst, len(out)))


# --------------------------------------------------------------------------------------
# 2. DrivAerML -- PhysicsNeMo-CFD 436/48 benchmark split, all 16 morph parameters
# --------------------------------------------------------------------------------------
def build_drivaerml():
    geo = _read_csv(os.path.join(CROSSBENCH, "drivaerml_geo.csv"))
    pcols = [c for c in geo[0] if c != "Run"]
    gmap = {r["Run"]: [float(r[c]) for c in pcols] for r in geo}
    tr = _read_csv(os.path.join(CROSSBENCH, "drivaerml_pn_train.csv"))
    va = _read_csv(os.path.join(CROSSBENCH, "drivaerml_pn_val.csv"))

    rows = []
    for tag, split_rows in (("train", tr), ("test", va)):
        for r in split_rows:
            rid = r["run_idx"]
            if rid not in gmap:
                continue
            rows.append([rid, tag, *gmap[rid], float(r["drag"])])
    header = ["run_id", "split"] + pcols + ["drag_force_N"]
    _write_csv(os.path.join(OUT, "drivaerml", "drivaerml_pn_436_48.csv"), header, rows)


# --------------------------------------------------------------------------------------
# 3. DrivAerNet++ -- official split, scored on the FULL 1154-design test set:
#    category tokens (rear/underbody/wheels/mirrors, one-hot, first level dropped,
#    levels fit on TRAIN only) + the 23 published parameters where they exist,
#    zero-filled for the unparametrised v1-fastback families. This is the
#    `category_params_where_published` row in null_travels.md -- a LOWER bound
#    on what published metadata supports (see that file, Sec 2.2).
# --------------------------------------------------------------------------------------
def build_drivaernet():
    params = _read_csv(os.path.join(CROSSBENCH, "drivaernet_params.csv"))
    cd_rows = _read_csv(os.path.join(CROSSBENCH, "drivaernet_cd.csv"))
    cd = {r["ID"]: float(r["Drag_Value"]) for r in cd_rows}
    pcols = [c for c in params[0]
             if c != "Experiment" and not c.startswith(("Average ", "Std "))]
    pvec = {r["Experiment"]: [float(r[c]) for c in pcols] for r in params}

    split = {}
    for tag in ("train", "test"):
        fname = {"train": "drivaernet_train_ids.txt", "test": "drivaernet_test_ids.txt"}[tag]
        with open(os.path.join(CROSSBENCH, fname), encoding="utf-8") as fh:
            split[tag] = [ln.strip() for ln in fh if ln.strip()]

    def cat_token(i):
        return "|".join(i.split("_")[:4])

    tr_all = [i for i in split["train"] if i in cd]
    te_all = [i for i in split["test"] if i in cd]
    levels_all = sorted(set(cat_token(i) for i in tr_all))  # fit on TRAIN only

    def onehot(i):
        tok = cat_token(i)
        j = levels_all.index(tok) if tok in levels_all else 0
        vec = [0.0] * (len(levels_all) - 1)
        if j > 0:
            vec[j - 1] = 1.0
        return vec

    def pfill(i):
        return pvec[i] if i in pvec else [0.0] * len(pcols)

    rows = []
    for tag, ids in (("train", tr_all), ("test", te_all)):
        for i in ids:
            rows.append([i, tag, *onehot(i), *pfill(i), cd[i]])
    header = (["design_id", "split"]
              + ["cat_%d" % k for k in range(len(levels_all) - 1)]
              + pcols + ["cd"])
    _write_csv(os.path.join(OUT, "drivaernet", "drivaernet_pp_5819_1154.csv"), header, rows)
    print("  category levels (train-fit): %d -> %d one-hot columns"
          % (len(levels_all), len(levels_all) - 1))


# --------------------------------------------------------------------------------------
# 4 & 5. AhmedML / WindsorML -- no published split, all published shape parameters,
#    constant-reference-area drag/lift (see null_travels.md Sec 2.3-2.4 for why
#    constant-area is the headline normalisation for each).
# --------------------------------------------------------------------------------------
def build_ashton(name, geo_file, force_file, id_geo, id_force, out_rel):
    geo = _read_csv(os.path.join(CROSSBENCH, geo_file))
    force = _read_csv(os.path.join(CROSSBENCH, force_file))
    pcols = [c for c in geo[0] if c != id_geo]
    gmap = {r[id_geo]: [float(r[c]) for c in pcols] for r in geo}
    rows = []
    for r in force:
        rid = r[id_force]
        if rid not in gmap:
            continue
        rows.append([rid, *gmap[rid], float(r["cd"]), float(r["cl"])])
    header = ["run_id"] + pcols + ["cd", "cl"]
    _write_csv(os.path.join(OUT, out_rel), header, rows)


def build_ahmedml():
    build_ashton("AhmedML", "ahmedml_geo.csv", "ahmedml_force.csv", "run", "run",
                 os.path.join("ahmedml", "ahmedml_500.csv"))


def build_windsorml():
    build_ashton("WindsorML", "windsorml_geo.csv", "windsorml_force.csv", "run", "run",
                 os.path.join("windsorml", "windsorml_355.csv"))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--airfrans-train-labels", action="store_true",
                     help="one-off: (re)generate the 800-case AirfRANS train label cache "
                          "from the local raw dataset (requires data/Dataset)")
    ap.add_argument("--root", default=os.path.join(REPO, "data", "Dataset"))
    args = ap.parse_args()

    if args.airfrans_train_labels:
        build_airfrans_train_labels(args.root)
        return 0

    missing = [n for n in ("ahmedml_geo.csv", "ahmedml_force.csv", "windsorml_geo.csv",
                            "windsorml_force.csv", "drivaerml_geo.csv",
                            "drivaerml_pn_train.csv", "drivaerml_pn_val.csv",
                            "drivaernet_params.csv", "drivaernet_cd.csv",
                            "drivaernet_train_ids.txt", "drivaernet_test_ids.txt")
               if not os.path.exists(os.path.join(CROSSBENCH, n))]
    if missing:
        print("missing cached crossbench metadata: %s\n"
              "run: .venv/Scripts/python.exe scripts/covariate_null_crossbench.py --download"
              % ", ".join(missing))
        return 2

    build_airfrans()
    build_drivaerml()
    build_drivaernet()
    build_ahmedml()
    build_windsorml()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
