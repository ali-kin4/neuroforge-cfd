"""Verify the mechanical submission requirements the target journal imposes.

TARGET VENUE, changed 2026-09-12: the Journal of Computational Science
(Elsevier, IF 4.0, hybrid with a free subscription route, single anonymized).
Computers & Fluids was withdrawn on an editorial conflict -- an author of the
benchmark this paper audits is editorially active there -- which is recorded in
docs/paper/review/journal_shortlist.md Sec. 4.

    abstract    <= 250 words
    keywords    1-7, and the guide discourages multi-word keywords joined by
                "and" or "of"
    highlights  3-5 bullets, each <= 85 characters INCLUDING spaces, in a
                separate file with "highlights" in its name
    manuscript  <= 12,000 words for a full-length article

THE LENGTH RULE IS NEW AND IT BINDS. Computers & Fluids set no length limit, so
this script never measured the body; JOCS caps a full-length article at 12,000
words, and the pre-rebuild manuscript was ~14,700. The body count below is
therefore a go/no-go, not a diagnostic. It is measured with the same strip_tex
semantics as the abstract, over abstract.tex + body.tex, which is the prose the
journal counts; LaTeX scaffolding (tabular rules, \\begin/\\end) contributes
almost nothing under those semantics, but table CELLS and captions do count,
which is the conservative direction.

Usage
-----
    python scripts/check_submission.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ABSTRACT_MAX_WORDS = 250
KEYWORDS_MIN, KEYWORDS_MAX = 1, 7
HIGHLIGHT_MAX_CHARS = 85
HIGHLIGHTS_MIN, HIGHLIGHTS_MAX = 3, 5
MANUSCRIPT_MAX_WORDS = 12000


def strip_tex(text: str) -> str:
    """Drop comments, control sequences and math punctuation for a word count."""
    body = "\n".join(l for l in text.split("\n") if not l.strip().startswith("%"))
    body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    for ch in "{}$~\\":
        body = body.replace(ch, " ")
    return body


def count_words(text: str) -> int:
    return len([w for w in strip_tex(text).split() if any(c.isalnum() for c in w)])


def check_abstract(root: str, fails: list[str]) -> None:
    path = os.path.join(root, "docs", "paper", "abstract.tex")
    n = count_words(open(path, encoding="utf-8").read())
    ok = n <= ABSTRACT_MAX_WORDS
    print(f"  abstract      {n:>4} words  (limit {ABSTRACT_MAX_WORDS})  "
          f"{'OK' if ok else 'OVER'}")
    if not ok:
        fails.append(f"abstract is {n} words, limit {ABSTRACT_MAX_WORDS}")


def check_manuscript_length(root: str, fails: list[str]) -> None:
    """Count the prose the journal counts: the abstract plus the shared body.

    Both build wrappers \\input the same two files, so this is venue-independent
    and cannot drift between the TMLR and the Elsevier build.
    """
    total = 0
    for name in ("abstract.tex", "body.tex"):
        path = os.path.join(root, "docs", "paper", name)
        n = count_words(open(path, encoding="utf-8").read())
        print(f"    {name:<14} {n:>6} words")
        total += n
    ok = total <= MANUSCRIPT_MAX_WORDS
    print(f"  manuscript  {total:>6} words  (limit {MANUSCRIPT_MAX_WORDS})  "
          f"{'OK' if ok else 'OVER'}")
    if not ok:
        fails.append(f"manuscript is {total} words, limit "
                     f"{MANUSCRIPT_MAX_WORDS} -- this is a go/no-go for JOCS")


def check_keywords(root: str, fails: list[str]) -> None:
    path = os.path.join(root, "docs", "paper", "neuroforge_cfd_elsevier.tex")
    src = open(path, encoding="utf-8").read()
    m = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", src, re.S)
    if not m:
        fails.append("no keyword block found")
        print("  keywords      MISSING")
        return
    block = "\n".join(l for l in m.group(1).split("\n")
                      if not l.strip().startswith("%"))
    words = [k.strip() for k in block.split(r"\sep") if k.strip()]
    ok = KEYWORDS_MIN <= len(words) <= KEYWORDS_MAX
    print(f"  keywords      {len(words):>4}        (limit {KEYWORDS_MAX})  "
          f"{'OK' if ok else 'OUT OF RANGE'}")
    if not ok:
        fails.append(f"{len(words)} keywords, allowed {KEYWORDS_MIN}-{KEYWORDS_MAX}")
    for w in words:
        if re.search(r"\b(and|of)\b", w):
            print(f"      ! '{w}' joins words with and/of, which the guide discourages")


def check_highlights(root: str, fails: list[str]) -> None:
    path = os.path.join(root, "docs", "paper", "submission", "highlights.txt")
    if not os.path.isfile(path):
        fails.append("highlights.txt missing (required at submission)")
        print("  highlights    MISSING -- the journal requires them at submission")
        return
    bullets = [l.strip()[2:].strip() for l in open(path, encoding="utf-8")
               if l.strip().startswith("- ")]
    n_ok = HIGHLIGHTS_MIN <= len(bullets) <= HIGHLIGHTS_MAX
    print(f"  highlights    {len(bullets):>4} bullets (allowed "
          f"{HIGHLIGHTS_MIN}-{HIGHLIGHTS_MAX})  {'OK' if n_ok else 'OUT OF RANGE'}")
    if not n_ok:
        fails.append(f"{len(bullets)} highlights, allowed "
                     f"{HIGHLIGHTS_MIN}-{HIGHLIGHTS_MAX}")
    for b in bullets:
        over = len(b) > HIGHLIGHT_MAX_CHARS
        print(f"      {'OVER' if over else 'ok  '} {len(b):>3} chars  {b}")
        if over:
            fails.append(f"highlight over {HIGHLIGHT_MAX_CHARS} chars: {b[:40]}...")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    print("Journal of Computational Science submission requirements\n")
    fails: list[str] = []
    check_abstract(args.root, fails)
    check_manuscript_length(args.root, fails)
    check_keywords(args.root, fails)
    check_highlights(args.root, fails)

    print("\nthe 12,000-word cap is the binding constraint at this venue; the "
          "count above\nis measured, not estimated.")
    if fails:
        print(f"\n{len(fails)} problem(s):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nall mechanical requirements satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
