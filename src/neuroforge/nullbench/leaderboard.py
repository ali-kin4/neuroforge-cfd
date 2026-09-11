"""Build the corrected leaderboard for a benchmark: every verified published
entry, beside the covariate null, beside the verdict.

Each :func:`build_leaderboard` call re-runs the harness from the shipped CSV
-- it never reads a cached number -- so the leaderboard and the code that
produced it cannot diverge. :func:`render_markdown` turns the JSON-able
record into the table format other authors can paste directly.
"""

from __future__ import annotations

from .benchmarks import REGISTRY, WINDSORML, windsor_implied_r2_floor
from .harness import run_null


def build_leaderboard(name: str, n_boot: int = 10000, seed: int | None = None) -> dict:
    """Run every target of benchmark ``name`` and assemble one leaderboard record."""
    cfg = REGISTRY[name]
    record = {
        "benchmark": cfg.name,
        "description": cfg.description,
        "verified_in": cfg.verified_in,
        "targets": {},
    }
    for tname, tgt in cfg.targets.items():
        published = list(tgt.published)
        if cfg is WINDSORML and tname == "cd":
            published = published + [windsor_implied_r2_floor()]
        table = cfg.load(tname)
        result = run_null(table, metric=tgt.metric, train_values=cfg.train_values,
                          test_values=cfg.test_values, folds=cfg.folds, n_boot=n_boot,
                          seed=cfg.seed if seed is None else seed, published=published)
        rec = result.to_dict()
        rec["target_note"] = tgt.note
        record["targets"][tname] = rec

    record["headline"] = _headline(record)
    return record


def _headline(record: dict) -> dict:
    """The single number a benchmark maintainer would quote: the drag/primary
    target's covariate-null fraction against the BEST verified published
    point estimate (bounds excluded -- see ``benchmarks.py``). Per-row
    fractions in ``targets.*.comparisons`` are the finer-grained reading;
    this is the one-line summary.
    """
    from .stats import FLOOR

    primary = "cd" if "cd" in record["targets"] else next(iter(record["targets"]))
    tgt = record["targets"][primary]
    floor = FLOOR[tgt["metric"]]
    candidates = [c for c in tgt["comparisons"]
                 if not c["is_bound"] and c["published"] is not None]
    bounds = [c for c in tgt["comparisons"] if c["is_bound"]]
    above_floor = [c for c in candidates if c["published"] > floor]

    if not candidates or not above_floor:
        if candidates and not above_floor:
            note = (f"every verified published entry for this target is at or below the "
                    f"metric's own floor ({floor}) -- no fraction is defined for any of "
                    f"them; see stats.py's floor convention. The published entries "
                    f"themselves not clearing the floor is the more important fact here.")
        else:
            note = "no verified published point-estimate baseline for this target"
        if bounds:
            note += (f" (a published BOUND exists -- {bounds[0]['model']}: "
                     f"{bounds[0]['verdict']} -- see the target's own comparisons; "
                     "a bound never yields a covariate-null fraction)")
        return {
            "target": primary,
            "metric": tgt["metric"],
            "protocol": tgt["protocol"],
            "null_point": tgt["out_of_sample"],
            "null_ci95": tgt["ci95"],
            "best_published_model": None,
            "best_published_value": None,
            "covariate_null_fraction": None,
            "note": note,
        }
    best = max(above_floor, key=lambda c: c["published"])
    return {
        "target": primary,
        "metric": tgt["metric"],
        "protocol": tgt["protocol"],
        "null_point": tgt["out_of_sample"],
        "null_ci95": tgt["ci95"],
        "best_published_model": best["model"],
        "best_published_value": best["published"],
        "covariate_null_fraction": best["covariate_null_fraction"],
        "verdict": best["verdict"],
        "note": "",
    }


def build_all_leaderboards(n_boot: int = 10000, seed: int | None = None) -> dict:
    return {name: build_leaderboard(name, n_boot=n_boot, seed=seed) for name in REGISTRY}


# --------------------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------------------
def _fmt(x, nd=4):
    if x is None:
        return "--"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render_markdown(record: dict) -> str:
    lines = []
    lines.append(f"## {record['benchmark']}")
    lines.append("")
    lines.append(record["description"])
    lines.append("")
    lines.append(f"Source verification: `{record['verified_in']}`")
    lines.append("")

    for tname, tgt in record["targets"].items():
        lines.append(f"### target: `{tname}` (metric: {tgt['metric']}, protocol: "
                     f"`{tgt['protocol']}`, n_fit={tgt['n_fit']}, n_score={tgt['n_score']})")
        lines.append("")
        if tgt.get("target_note"):
            lines.append(f"> {tgt['target_note']}")
            lines.append("")
        lines.append(f"Null out-of-sample: **{_fmt(tgt['out_of_sample'])}** "
                     f"[{_fmt(tgt['ci95'][0])}, {_fmt(tgt['ci95'][1])}] "
                     f"(95% case-level bootstrap, n_boot={tgt['n_boot_finite']})")
        lines.append("")
        if not tgt["comparisons"]:
            lines.append("_No published entries verified for this target._")
            lines.append("")
            continue
        lines.append("| model | published | published std | null | null CI95 | "
                     "covariate-null fraction | flags | verdict | source |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in tgt["comparisons"]:
            cnf = c["covariate_null_fraction"]
            cnf_str = "--" if cnf is None else _fmt(cnf["covariate_null_fraction"])
            flags = ", ".join(cnf["flags"]) if cnf and cnf["flags"] else (
                "bound" if c["is_bound"] else "")
            lines.append(
                f"| {c['model']} | {_fmt(c['published'])}"
                f"{'*' if c['is_bound'] else ''} | {_fmt(c['published_std'])} | "
                f"{_fmt(tgt['out_of_sample'])} | [{_fmt(tgt['ci95'][0])}, "
                f"{_fmt(tgt['ci95'][1])}] | {cnf_str} | {flags} | {c['verdict']} | "
                f"{c['published_source']} |")
        lines.append("")

    h = record["headline"]
    lines.append(f"**Headline** (target `{h['target']}`, metric {h['metric']}): "
                 f"null {_fmt(h['null_point'])} [{_fmt(h['null_ci95'][0])}, "
                 f"{_fmt(h['null_ci95'][1])}] vs best verified published "
                 f"`{h['best_published_model']}` = {_fmt(h.get('best_published_value'))} "
                 f"-> covariate-null fraction "
                 f"{_fmt((h.get('covariate_null_fraction') or {}).get('covariate_null_fraction')) if h.get('covariate_null_fraction') else '--'}"
                 f". {h.get('note', '')}")
    lines.append("")
    return "\n".join(lines)


def render_markdown_all(records: dict) -> str:
    parts = ["# NullBench — corrected leaderboards", "",
            "Machine-readable form: the sibling `.json` files in this directory. "
            "Regenerate both with `neuroforge nullbench leaderboard --all`. See "
            "`docs/NULLBENCH.md` for the statistic's definition.", ""]
    for name in records:
        parts.append(render_markdown(records[name]))
    return "\n".join(parts)
