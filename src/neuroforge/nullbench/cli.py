"""``neuroforge nullbench`` -- run the covariate-null harness from the command line.

Three modes:

``run``
    Generic mode: point it at any CSV of ``(case id, parameters..., label[, split])``.
``bench``
    Run one of the five shipped worked examples (``airfrans``, ``drivaerml``,
    ``drivaernet``, ``ahmedml``, ``windsorml``).
``leaderboard``
    Rebuild the corrected leaderboard (JSON + markdown) for one benchmark
    (``--name``), all five under their own protocols (``--all``, which also
    writes the harmonised table), or just the cross-benchmark-comparable
    harmonised table (``--harmonised``; see ``neuroforge.nullbench.harmonised``).

Every JSON output is written with ``newline="\\n"`` -- Windows' default text
mode would emit CRLF, which breaks the SHA-256 hashes in ``results/MANIFEST.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def _cmd_run(args: argparse.Namespace) -> int:
    from .harness import PublishedEntry, permutation_check, run_null
    from .io import load_table

    feature_cols = args.feature_cols.split(",") if args.feature_cols else None
    table = load_table(args.csv, id_col=args.id_col, label_col=args.label_col,
                       feature_cols=feature_cols, split_col=args.split_col)

    published = []
    if args.published:
        with open(args.published, encoding="utf-8") as fh:
            entries = json.load(fh)
        for e in entries:
            published.append(PublishedEntry(
                name=e["name"], value=e.get("value"), std=e.get("std"),
                source=e.get("source", ""), note=e.get("note", ""),
                is_bound=e.get("is_bound", False), bound_kind=e.get("bound_kind", "upper"),
                precision=e.get("precision")))

    result = run_null(table, metric=args.metric,
                      train_values=tuple(args.train_values.split(",")),
                      test_values=tuple(args.test_values.split(",")),
                      folds=args.folds, n_boot=args.boot, seed=args.seed,
                      published=published)
    out = result.to_dict()
    out["split_used"] = "official_split (from --split-col)" if table.has_split() else \
        f"kfold_oos (no split column; k={args.folds})"

    if args.permute_check:
        out["permutation_check"] = permutation_check(
            table, metric=args.metric, train_values=tuple(args.train_values.split(",")),
            test_values=tuple(args.test_values.split(",")), folds=args.folds, seed=args.seed + 1)

    print(json.dumps(out, indent=2))
    if args.out:
        _write_json(args.out, out)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .benchmarks import REGISTRY, run_benchmark

    if args.name not in REGISTRY:
        print(f"unknown benchmark {args.name!r}; choices: {list(REGISTRY)}", file=sys.stderr)
        return 2
    cfg = REGISTRY[args.name]
    target = args.target or next(iter(cfg.targets))
    result = run_benchmark(cfg, target, n_boot=args.boot)
    out = result.to_dict()
    print(json.dumps(out, indent=2))
    if args.out:
        _write_json(args.out, out)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


def _write_harmonised(out_dir: str, boot: int) -> None:
    from .leaderboard import build_harmonised_table, render_harmonised_markdown

    table = build_harmonised_table(n_boot=boot)
    _write_json(os.path.join(out_dir, "harmonised.json"), table)
    with open(os.path.join(out_dir, "harmonised.md"), "w", encoding="utf-8",
             newline="\n") as fh:
        fh.write(render_harmonised_markdown(table))
    print(f"wrote harmonised cross-benchmark table to {out_dir}/harmonised.{{json,md}}")


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    from .leaderboard import (
        build_all_leaderboards,
        build_leaderboard,
        render_markdown,
        render_markdown_all,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    if args.harmonised:
        _write_harmonised(args.out_dir, args.boot)
        return 0

    if args.all:
        records = build_all_leaderboards(n_boot=args.boot)
        for name, rec in records.items():
            _write_json(os.path.join(args.out_dir, f"{name}.json"), rec)
            with open(os.path.join(args.out_dir, f"{name}.md"), "w",
                     encoding="utf-8", newline="\n") as fh:
                fh.write(render_markdown(rec))
        with open(os.path.join(args.out_dir, "ALL.md"), "w",
                 encoding="utf-8", newline="\n") as fh:
            fh.write(render_markdown_all(records))
        print(f"wrote {len(records)} leaderboards to {args.out_dir}")
        # `--all` means "every corrected leaderboard", which includes the
        # harmonised cross-benchmark view -- see harmonised.py for why it is a
        # separate table rather than a column bolted onto the per-benchmark ones.
        _write_harmonised(args.out_dir, args.boot)
    else:
        rec = build_leaderboard(args.name, n_boot=args.boot)
        _write_json(os.path.join(args.out_dir, f"{args.name}.json"), rec)
        with open(os.path.join(args.out_dir, f"{args.name}.md"), "w",
                 encoding="utf-8", newline="\n") as fh:
            fh.write(render_markdown(rec))
        print(json.dumps(rec["headline"], indent=2))
        print(f"\nwrote {args.out_dir}/{args.name}.{{json,md}}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="neuroforge nullbench", description=__doc__)
    sub = ap.add_subparsers(dest="mode", metavar="<mode>")

    p_run = sub.add_parser("run", help="run on any (id, params..., label[, split]) CSV")
    p_run.add_argument("--csv", required=True)
    p_run.add_argument("--id-col", required=True)
    p_run.add_argument("--label-col", required=True)
    p_run.add_argument("--feature-cols", default=None,
                       help="comma-separated; default: every non-id/label/split column")
    p_run.add_argument("--split-col", default=None,
                       help="omit for K-fold out-of-sample")
    p_run.add_argument("--train-values", default="train")
    p_run.add_argument("--test-values", default="test")
    p_run.add_argument("--metric", choices=["r2", "spearman"], default="r2")
    p_run.add_argument("--folds", type=int, default=10)
    p_run.add_argument("--boot", type=int, default=10000)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--published", default=None,
                       help="JSON list of {name, value, std, source, is_bound, bound_kind, "
                            "precision} -- precision (decimal digits) drives the "
                            "rounding-sensitivity check on the published-relative ratio")
    p_run.add_argument("--permute-check", action="store_true",
                       help="also fit on label-shuffled training data (should land at/below the floor)")
    p_run.add_argument("--out", default=None)
    p_run.set_defaults(func=_cmd_run)

    p_bench = sub.add_parser("bench", help="run a shipped worked example")
    p_bench.add_argument("--name", required=True,
                         choices=["airfrans", "drivaerml", "drivaernet", "ahmedml", "windsorml"])
    p_bench.add_argument("--target", default=None, help="default: the config's first target")
    p_bench.add_argument("--boot", type=int, default=10000)
    p_bench.add_argument("--out", default=None)
    p_bench.set_defaults(func=_cmd_bench)

    p_lb = sub.add_parser("leaderboard", help="rebuild the corrected leaderboard(s)")
    p_lb.add_argument("--name", default=None,
                      choices=["airfrans", "drivaerml", "drivaernet", "ahmedml", "windsorml"])
    p_lb.add_argument("--all", action="store_true",
                      help="every per-benchmark leaderboard PLUS the harmonised table")
    p_lb.add_argument("--harmonised", action="store_true",
                      help="only the cross-benchmark-comparable harmonised table "
                           "(see neuroforge.nullbench.harmonised)")
    p_lb.add_argument("--boot", type=int, default=10000)
    p_lb.add_argument("--out-dir", default="results/nullbench/leaderboards")
    p_lb.set_defaults(func=_cmd_leaderboard)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "mode", None):
        ap.print_help()
        return 0
    if args.mode == "leaderboard" and not args.all and not args.name and not args.harmonised:
        print("leaderboard mode needs --name NAME, --all, or --harmonised", file=sys.stderr)
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
