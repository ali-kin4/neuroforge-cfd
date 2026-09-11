"""Build the corrected leaderboard for a benchmark: every verified published
entry, beside the metadata null, beside the verdict -- PLUS the separate
harmonised cross-benchmark view (see ``harmonised.py`` for why the two must
never be merged into one table).

Each :func:`build_leaderboard` / :func:`build_harmonised_table` call re-runs
the harness from the shipped CSV -- it never reads a cached number -- so the
leaderboard and the code that produced it cannot diverge. :func:`render_markdown`
turns the JSON-able record into the table format other authors can paste
directly.
"""

from __future__ import annotations

from .benchmarks import REGISTRY, WINDSORML, windsor_implied_r2_floor
from .harmonised import run_all_harmonised
from .stats import FLOOR


def build_leaderboard(name: str, n_boot: int = 10000, seed: int | None = None) -> dict:
    """Run every target of benchmark ``name`` (under its OWN protocol) and
    assemble one leaderboard record."""
    from .harness import run_null

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
    target's ``metadata_null_{metric}`` (the headline statistic; see
    stats.py) alongside the published-relative ratio against the BEST
    verified published point estimate (bounds excluded -- see
    ``benchmarks.py``). Per-row ratios in ``targets.*.comparisons`` are the
    finer-grained reading; this is the one-line summary.
    """
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
                    f"metric's own floor ({floor}) -- no ratio is defined for any of "
                    f"them; see stats.py's floor convention. The published entries "
                    f"themselves not clearing the floor is the more important fact here.")
        else:
            note = "no verified published point-estimate baseline for this target"
        if bounds:
            note += (f" (a published BOUND exists -- {bounds[0]['model']}: "
                     f"{bounds[0]['verdict']} -- see the target's own comparisons; "
                     "a bound never yields a published-relative ratio)")
        return {
            "target": primary,
            "metric": tgt["metric"],
            "protocol": tgt["protocol"],
            "metadata_null_r2" if tgt["metric"] == "r2" else "metadata_null_spearman":
                tgt["out_of_sample"],
            "null_point": tgt["out_of_sample"],
            "null_ci95": tgt["ci95"],
            "best_published_model": None,
            "best_published_value": None,
            "published_relative_ratio": None,
            "note": note,
        }
    best = max(above_floor, key=lambda c: c["published"])
    return {
        "target": primary,
        "metric": tgt["metric"],
        "protocol": tgt["protocol"],
        "metadata_null_r2" if tgt["metric"] == "r2" else "metadata_null_spearman":
            tgt["out_of_sample"],
        "null_point": tgt["out_of_sample"],
        "null_ci95": tgt["ci95"],
        "best_published_model": best["model"],
        "best_published_value": best["published"],
        "published_relative_ratio": best["published_relative_ratio"],
        "verdict": best["verdict"],
        "note": "",
    }


def build_all_leaderboards(n_boot: int = 10000, seed: int | None = None) -> dict:
    return {name: build_leaderboard(name, n_boot=n_boot, seed=seed) for name in REGISTRY}


def build_harmonised_table(n_boot: int = 10000) -> dict:
    """The cross-benchmark-comparable view: one protocol, all five benchmarks,
    the linear metadata null alongside the sourced flexible ceiling. See
    ``harmonised.py``'s module docstring for the full rationale."""
    from .harmonised import PROTOCOL, PROTOCOL_SOURCE

    results = run_all_harmonised(n_boot=n_boot)
    return {
        "protocol": PROTOCOL,
        "protocol_source": PROTOCOL_SOURCE,
        "note": ("Cross-benchmark-comparable by construction: identical protocol, "
                "identical metric, applied to all five. NOT the per-benchmark worked "
                "examples in the sibling leaderboard files, which use each "
                "benchmark's own split/metric and are not comparable to each other "
                "or to this table -- see benchmarks.py and harmonised.py."),
        "benchmarks": {name: r.to_dict() for name, r in results.items()},
        "ranking_by_metadata_null_r2": sorted(
            ((r.benchmark, r.metadata_null_r2) for r in results.values()),
            key=lambda t: t[1], reverse=True),
    }


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
        metadata_key = f"metadata_null_{tgt['metric']}"
        lines.append(f"{metadata_key}: **{_fmt(tgt[metadata_key])}** "
                     f"[{_fmt(tgt['ci95'][0])}, {_fmt(tgt['ci95'][1])}] "
                     f"(95% case-level bootstrap, n_boot={tgt['n_boot_finite']}); "
                     f"metadata_null_mse: {_fmt(tgt['metadata_null_mse'], 3)}")
        lines.append("")
        if not tgt["comparisons"]:
            lines.append("_No published entries verified for this target._")
            lines.append("")
            continue
        lines.append("| model | published | published std | metadata null | null CI95 | "
                     "published-relative ratio | flags | verdict | source |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in tgt["comparisons"]:
            ratio = c["published_relative_ratio"]
            ratio_str = "--" if ratio is None else _fmt(ratio["published_relative_ratio"])
            flags = ", ".join(ratio["flags"]) if ratio and ratio["flags"] else (
                "bound" if c["is_bound"] else "")
            lines.append(
                f"| {c['model']} | {_fmt(c['published'])}"
                f"{'*' if c['is_bound'] else ''} | {_fmt(c['published_std'])} | "
                f"{_fmt(tgt[metadata_key])} | [{_fmt(tgt['ci95'][0])}, "
                f"{_fmt(tgt['ci95'][1])}] | {ratio_str} | {flags} | {c['verdict']} | "
                f"{c['published_source']} |")
        lines.append("")

    h = record["headline"]
    ratio = h.get("published_relative_ratio")
    ratio_str = _fmt(ratio["published_relative_ratio"]) if ratio else "--"
    lines.append(f"**Headline** (target `{h['target']}`, metric {h['metric']}): "
                 f"metadata null {_fmt(h['null_point'])} [{_fmt(h['null_ci95'][0])}, "
                 f"{_fmt(h['null_ci95'][1])}] vs best verified published "
                 f"`{h['best_published_model']}` = {_fmt(h.get('best_published_value'))} "
                 f"-> published-relative ratio {ratio_str}. {h.get('note', '')}")
    lines.append("")
    return "\n".join(lines)


def render_markdown_all(records: dict) -> str:
    parts = ["# NullBench — corrected leaderboards (per-benchmark, own protocol)", "",
            "Machine-readable form: the sibling `.json` files in this directory. "
            "Regenerate with `neuroforge nullbench leaderboard --all`. See "
            "`docs/NULLBENCH.md` for the statistic's definition, and "
            "`harmonised.md` in this directory for the SEPARATE cross-benchmark "
            "view -- these per-benchmark tables are not comparable to each other; "
            "see benchmarks.py's module docstring.", ""]
    for name in records:
        parts.append(render_markdown(records[name]))
    return "\n".join(parts)


def render_harmonised_markdown(table: dict) -> str:
    lines = ["# NullBench — harmonised cross-benchmark view", "",
            f"Protocol: **{table['protocol']}** (`{table['protocol_source']}`)", "",
            table["note"], "",
            "| benchmark | n | metadata_null_r2 | 95% CI | flexible ceiling (sourced) | "
            "linearity share (sourced) |",
            "|---|---|---|---|---|---|"]
    for b in table["benchmarks"].values():
        lines.append(
            f"| {b['benchmark']} | {b['n_cases']} | {_fmt(b['metadata_null_r2'])} | "
            f"[{_fmt(b['metadata_null_r2_ci95'][0])}, {_fmt(b['metadata_null_r2_ci95'][1])}] | "
            f"{_fmt(b['flexible_ceiling_r2_sourced'])} | "
            f"{_fmt(b['linearity_share_sourced'])} |")
    lines.append("")
    lines.append("Ranking by metadata_null_r2 (highest to lowest): " +
                 ", ".join(f"{n} ({_fmt(v)})" for n, v in table["ranking_by_metadata_null_r2"]))
    lines.append("")
    lines.append(f"> {table['benchmarks'][next(iter(table['benchmarks']))]['published_comparison_note']}")
    lines.append("")
    lines.append("Flexible-ceiling values are READ from "
                 f"`{next(iter(table['benchmarks'].values()))['flexible_ceiling_source']}`, "
                 "not recomputed by this harness -- see `harmonised.py`'s module docstring.")
    lines.append("")
    return "\n".join(lines)
