"""Assemble the arXiv source tarball for a replacement of arXiv:2607.10333.

arXiv compiles from a flat-ish archive, so the figures move from
``results/figures/`` (where the repository keeps them, alongside the scripts that
produce them) to ``figures/`` inside the archive, and ``body.tex``'s
``\\includegraphics`` paths are rewritten to match. The only other change is that
full-line ``%`` comments are dropped from the staged ``.tex`` files: they are working
notes, arXiv publishes the source, and TeX discards such a line together with its
end-of-line, so removing it cannot change the output. The staged build must reproduce
the in-place build's every label, number and page, which is what shows the preprint
cannot drift from the paper.

Two things this script enforces because getting either wrong wastes a
replacement slot:

* **The ``.bbl`` ships.** arXiv's "delete .bbl" box is pre-ticked on the Review
  Files step and must be unticked every time; letting arXiv re-run BibTeX is
  untested for this manuscript. The staged tarball therefore contains the
  ``.bbl`` built here, and the script verifies it is newer than ``refs.bib``.
* **Only figures the manuscript actually includes** are staged, so an unused
  file cannot silently change what arXiv builds. No ``sections/`` file is staged:
  none is ``\\input`` by either build.

The staged tree is compiled before the tarball is written, so a package that
does not build cannot be produced.

Usage
-----
    python scripts/build_arxiv_package.py
    python scripts/build_arxiv_package.py --version v5 --out docs/paper/submission/arxiv_v5
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile

TEX_ROOT = os.path.join("docs", "paper")
FIG_SRC = os.path.join("results", "figures")
MAIN = "neuroforge_cfd"          # the TMLR build is the preprint build
TOP_LEVEL = ["abstract.tex", "body.tex", "preamble.tex", f"{MAIN}.tex",
             f"{MAIN}.bbl", "refs.bib", "tmlr.sty", "tmlr.bst", "fancyhdr.sty"]
SECTIONS: list[str] = []
OLD_FIG_PREFIX = "../../results/figures/"
NEW_FIG_PREFIX = "figures/"


def strip_comment_lines(text: str) -> str:
    """Drop lines that are entirely a comment (TeX ignores them with their newline)."""
    return "".join(l for l in text.splitlines(keepends=True)
                   if not re.match(r"[ \t]*%", l))


def labels(aux_path: str) -> dict:
    aux = open(aux_path, encoding="utf-8", errors="replace").read()
    return {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(
        r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", aux)}


def figures_used(body: str) -> list[str]:
    """Basenames of every figure the manuscript actually includes."""
    hits = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", body)
    return sorted({os.path.basename(h) for h in hits})


def stage(root: str, out: str) -> tuple[str, list[str]]:
    stage_dir = os.path.join(out, "stage")
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(os.path.join(stage_dir, "figures"), exist_ok=True)

    for name in TOP_LEVEL:
        shutil.copy2(os.path.join(root, TEX_ROOT, name),
                     os.path.join(stage_dir, name))
        if name.endswith(".tex"):
            path = os.path.join(stage_dir, name)
            text = open(path, encoding="utf-8").read()
            open(path, "w", encoding="utf-8", newline="\n").write(strip_comment_lines(text))
    for rel in SECTIONS:
        shutil.copy2(os.path.join(root, TEX_ROOT, rel),
                     os.path.join(stage_dir, rel))

    body_path = os.path.join(stage_dir, "body.tex")
    body = open(body_path, encoding="utf-8").read()
    used = figures_used(body)
    body = body.replace(OLD_FIG_PREFIX, NEW_FIG_PREFIX)
    open(body_path, "w", encoding="utf-8", newline="\n").write(body)

    for fig in used:
        src = os.path.join(root, FIG_SRC, fig)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"figure referenced but missing: {src}")
        shutil.copy2(src, os.path.join(stage_dir, "figures", fig))
    return stage_dir, used


def build(stage_dir: str) -> None:
    """Compile the staged tree exactly as arXiv would, and fail loudly."""
    for i in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-halt-on-error", f"{MAIN}.tex"],
                           cwd=stage_dir, capture_output=True, text=True)
        if r.returncode != 0:
            tail = "\n".join(r.stdout.strip().split("\n")[-25:])
            raise SystemExit(f"staged build failed on pass {i + 1}:\n{tail}")
    log = open(os.path.join(stage_dir, f"{MAIN}.log"), encoding="utf-8",
               errors="replace").read()
    for bad in ("Undefined control sequence", "Undefined citation",
                "There were undefined references"):
        if bad in log:
            raise SystemExit(f"staged build reports: {bad}")
    for bad in ("Overfull", "Underfull", "LaTeX Warning"):
        if bad in log:
            raise SystemExit(f"staged build reports: {bad}")
    pages = re.search(r"Output written on .*?\((\d+) pages", log)
    print(f"  staged build OK: {pages.group(1) if pages else '?'} pages")


def make_tarball(stage_dir: str, out: str, version: str) -> str:
    """Archive with the source at the top level, as arXiv expects."""
    path = os.path.join(out, f"arxiv_{version}_source.tar.gz")
    keep = set(TOP_LEVEL) | {os.path.basename(s) for s in SECTIONS}
    with tarfile.open(path, "w:gz") as tar:
        for name in TOP_LEVEL:
            tar.add(os.path.join(stage_dir, name), arcname=name)
        for rel in SECTIONS:
            tar.add(os.path.join(stage_dir, rel), arcname=rel.replace(os.sep, "/"))
        figdir = os.path.join(stage_dir, "figures")
        for fig in sorted(os.listdir(figdir)):
            tar.add(os.path.join(figdir, fig), arcname=f"figures/{fig}")
    del keep
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--version", default="v5")
    ap.add_argument("--out", default=os.path.join("docs", "paper", "submission",
                                                  "arxiv_v5"))
    args = ap.parse_args(argv)

    out = os.path.join(args.root, args.out)
    os.makedirs(out, exist_ok=True)

    bbl = os.path.join(args.root, TEX_ROOT, f"{MAIN}.bbl")
    bib = os.path.join(args.root, TEX_ROOT, "refs.bib")
    if os.path.getmtime(bbl) < os.path.getmtime(bib):
        print("  ! the .bbl is older than refs.bib -- rebuild the paper "
              "(pdflatex, bibtex, pdflatex x2) before packaging")
        return 1

    print(f"staging {args.version} from {TEX_ROOT}")
    stage_dir, used = stage(args.root, out)
    print(f"  {len(used)} figures: {', '.join(used)}")
    build(stage_dir)
    ref = os.path.join(args.root, TEX_ROOT, f"{MAIN}.aux")
    if labels(os.path.join(stage_dir, f"{MAIN}.aux")) != labels(ref):
        print("  ! the staged build's labels differ from the in-place build's -- "
              "rebuild the paper in place, then package again")
        return 1
    print(f"  labels and pages identical to the in-place build ({len(labels(ref))} labels)")
    path = make_tarball(stage_dir, out, args.version)
    size = os.path.getsize(path) / 1024
    print(f"  wrote {path} ({size:.0f} KB)")

    pdf = os.path.join(stage_dir, f"{MAIN}.pdf")
    final = os.path.join(out, f"{MAIN}_{args.version}.pdf")
    shutil.copy2(pdf, final)
    print(f"  wrote {final}")
    print("\nremember: UNTICK arXiv's pre-ticked \"delete .bbl\" box on Review Files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
