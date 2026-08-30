"""Render ``docs/paper2/DRAFT.md`` to a PDF, reproducibly.

The draft is written in Markdown because that is what stays honest under heavy
revision -- a table you can edit in place is a table you keep up to date. But a
paper has to be *read* as a paper, so this converts it to LaTeX and compiles it.

Deliberately a converter and not a one-off ``.tex`` file: the draft changes
several times a session, and a hand-maintained LaTeX copy would drift from it
within a day. **The Markdown is the source of truth**; this script is the only
thing that may produce the PDF, so the two cannot disagree.

Scope is exactly what the draft uses -- headings, tables, blockquotes, lists,
bold/italic/code, and the fifteen non-ASCII characters it contains (checked, not
assumed). Anything else it does not handle it escapes rather than guesses at.

Usage
-----
    python scripts/build_paper2_pdf.py
    python scripts/build_paper2_pdf.py --keep-tex     # leave the .tex to inspect
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUTHORS = (
    r"Ali Jabbary\thanks{Corresponding author: \texttt{st\_a.jabbary@urmia.ac.ir}. "
    r"ORCID 0000-0003-0573-6909.}\\"
    "\n"
    r"\small Department of Mechanical Engineering, Urmia University, Urmia, Iran"
    "\n\\and\n"
    r"Kasra Ghanavati\\"
    "\n"
    r"\small School of Computing and Mathematical Sciences,\\"
    "\n"
    r"\small University of Greenwich, London, UK"
)

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{framed}
\usepackage{textcomp}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}

\definecolor{shadecolor}{RGB}{244,243,239}
\setlength{\parskip}{0.55em}
\setlength{\parindent}{0pt}
\setlist{nosep,leftmargin=1.4em}
\titlespacing*{\section}{0pt}{1.4em}{0.5em}
\titlespacing*{\subsection}{0pt}{1.1em}{0.4em}
\renewcommand{\arraystretch}{1.15}

% Tables in this draft are narrow and numerous; keep them upright and small.
\newcommand{\tabfont}{\small}
"""


def protect_code(text: str, store: list[str]) -> str:
    """Pull `code spans` out before escaping, so their contents stay literal."""
    def sub(m):
        store.append(m.group(1))
        return f"\x00{len(store) - 1}\x00"
    return re.sub(r"`([^`]+)`", sub, text)


def restore_code(text: str, store: list[str]) -> str:
    def sub(m):
        body = store[int(m.group(1))]
        for a, b in (("\\", r"\textbackslash{}"), ("{", r"\{"), ("}", r"\}"),
                     ("_", r"\_"), ("^", r"\^{}"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#"), ("~", r"\textasciitilde{}")):
            body = body.replace(a, b)
        return r"\texttt{" + body + "}"
    return re.sub("\x00(\\d+)\x00", sub, text)


# The exact non-ASCII inventory of the draft, checked with a character census
# rather than guessed. Superscript runs are handled before the singles so that
# "10^-5" written as digits plus U+207B collapses into one math group.
SUPERS = {"\u00b2": "2", "\u00b3": "3", "\u2075": "5", "\u2076": "6",
          "\u207b": "-"}
SINGLES = [
    ("\u2014", "---"), ("\u2013", "--"), ("\u00a7", r"\S"),
    ("\u2212", "$-$"), ("\u00b0", r"\textdegree{}"), ("\u00d7", r"$\times$"),
    ("\u2192", r"$\rightarrow$"), ("\u2264", r"$\le$"), ("\u2265", r"$\ge$"),
    ("\u2248", r"$\approx$"), ("\u00b7", r"$\cdot$"),
]


def unicode_to_tex(text: str) -> str:
    def sup(m):
        return "$^{" + "".join(SUPERS[c] for c in m.group(0)) + "}$"
    text = re.sub("[" + "".join(SUPERS) + "]+", sup, text)
    for a, b in SINGLES:
        text = text.replace(a, b)
    return text


def escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\^{}")):
        text = text.replace(a, b)
    return text


def inline(text: str) -> str:
    """One line of Markdown body text to LaTeX."""
    store: list[str] = []
    text = protect_code(text, store)
    # "128^2" with an ASCII caret is maths, not a circumflex. Marked before
    # escaping and restored after, so escape() cannot eat the caret.
    text = re.sub(r"(?<=\d)\^(\d+)", r"@SUP\g<1>SUP@", text)
    text = escape(text)
    text = re.sub(r"@SUP(\d+)SUP@", r"$^{\g<1>}$", text)
    text = unicode_to_tex(text)
    # Links before emphasis: the label may itself contain ** markers.
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                  lambda m: r"\href{" + m.group(2).replace("%", r"\%")
                            + "}{" + m.group(1) + "}", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\*\w])\*([^*]+?)\*(?!\*)", r"\\emph{\1}", text)
    return restore_code(text, store)


def table(rows: list[str]) -> str:
    """A Markdown pipe table to a booktabs tabular."""
    def cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header, spec, *body = rows
    align = "".join("r" if c.endswith(":") and not c.startswith(":")
                    else ("c" if c.startswith(":") and c.endswith(":") else "l")
                    for c in cells(spec))
    # p-column for the first column of wide tables keeps long labels from
    # pushing the table off the page; every draft table is label-then-numbers.
    if len(align) >= 4:
        align = "p{0.20\\textwidth}" + align[1:]
    out = [r"\begin{center}\tabfont",
           # Shrink only if the natural width overflows; a narrow table drawn
           # at \textwidth would be blown up instead. The seven-column band
           # table ran off the right margin without this.
           r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
           r"\begin{tabular}{" + align + "}", r"\toprule",
           " & ".join(inline(c) for c in cells(header)) + r" \\", r"\midrule"]
    for line in body:
        out.append(" & ".join(inline(c) for c in cells(line)) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{center}"]
    return "\n".join(out)


def unwrap(lines: list[str]) -> list[str]:
    """Join hard-wrapped body lines into one logical line each.

    The draft is wrapped at 80 columns, so ``**a bold phrase**`` routinely
    straddles a newline. Converting line by line leaves the literal asterisks in
    the PDF -- which is what the first build of this script did. Structural
    lines (headings, tables, quotes, rules, list openers) are never joined.
    """
    def structural(s: str) -> bool:
        t = s.lstrip()
        return (not t or t.startswith(("#", "|", ">", "---"))
                or re.match(r"^(\d+\.|[-*])\s", t) is not None)

    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        if structural(line):
            out.append(line)
            i += 1
            # A list item or plain paragraph may continue on the next lines.
            if re.match(r"^\s*(\d+\.|[-*])\s", line):
                while i < n and lines[i].strip() and not structural(lines[i]):
                    out[-1] += " " + lines[i].strip()
                    i += 1
            continue
        buf = [line.strip()]
        i += 1
        while i < n and lines[i].strip() and not structural(lines[i]):
            buf.append(lines[i].strip())
            i += 1
        out.append(" ".join(buf))
    return out


def convert(md: str) -> str:
    lines = unwrap(md.split("\n"))
    out: list[str] = []
    i, n = 0, len(lines)
    list_env: str | None = None

    def close_list():
        nonlocal list_env
        if list_env:
            out.append(r"\end{" + list_env + "}")
            list_env = None

    while i < n:
        line = lines[i]

        if line.startswith("# "):          # the title is set by \maketitle
            i += 1
            continue
        if line.strip() == "---":
            close_list(); i += 1
            continue
        if not line.strip():
            close_list(); out.append(""); i += 1
            continue

        if line.startswith("### "):
            close_list()
            out.append(r"\subsection*{" + inline(line[4:]) + "}")
            i += 1
            continue
        if line.startswith("## "):
            close_list()
            out.append(r"\section*{" + inline(line[3:]) + "}")
            i += 1
            continue

        if line.lstrip().startswith("|") and i + 1 < n and set(
                lines[i + 1].replace("|", "").replace(" ", "")) <= set("-:"):
            close_list()
            block = []
            while i < n and lines[i].lstrip().startswith("|"):
                block.append(lines[i]); i += 1
            out.append(table(block))
            continue

        if line.startswith(">"):
            close_list()
            block = []
            while i < n and lines[i].startswith(">"):
                block.append(lines[i].lstrip(">").strip()); i += 1
            text = " ".join(block)
            out.append(r"\begin{shaded}\noindent " + inline(text)
                       + r"\end{shaded}")
            # A blockquote that names a rendered figure gets the figure. The
            # draft describes Figures 1 and 2 in prose and points at the PNG;
            # a paper has to actually show them.
            for path in re.findall(r"results/[\w./-]+\.png", text):
                if os.path.isfile(os.path.join(HERE, path)):
                    out.append(r"\begin{center}\includegraphics[width=\textwidth]{"
                               + path + r"}\end{center}")
            continue

        m = re.match(r"^(\d+)\.\s+(.*)", line)
        if m:
            if list_env != "enumerate":
                close_list(); out.append(r"\begin{enumerate}"); list_env = "enumerate"
            out.append(r"\item " + inline(m.group(2))); i += 1
            continue
        m = re.match(r"^[-*]\s+(.*)", line)
        if m:
            if list_env != "itemize":
                close_list(); out.append(r"\begin{itemize}"); list_env = "itemize"
            out.append(r"\item " + inline(m.group(1))); i += 1
            continue

        close_list()
        out.append(inline(line)); i += 1

    close_list()
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=os.path.join("docs", "paper2", "DRAFT.md"))
    ap.add_argument("--out", default=os.path.join("docs", "paper2", "paper2.pdf"))
    ap.add_argument("--keep-tex", action="store_true")
    args = ap.parse_args(argv)

    src = os.path.join(HERE, args.src)
    if not os.path.isfile(src):
        print(f"missing {args.src}")
        return 1
    md = open(src, encoding="utf-8").read()

    # Escape each half *before* inserting the LaTeX, or the line break and
    # \large are escaped along with the prose -- which is what build one did.
    raw = md.split("\n", 1)[0].lstrip("# ").strip()
    head, sep, tail = raw.partition(":")
    title = unicode_to_tex(escape(head.strip()))
    if sep:
        title += ":" + r"\\[0.35em] \large " + unicode_to_tex(escape(tail.strip()))

    tex = "\n".join([
        PREAMBLE,
        r"\title{\bfseries " + title + "}",
        r"\author{" + AUTHORS + "}",
        r"\date{\today\\[0.4em]\small Draft --- not submitted}",
        r"\begin{document}",
        r"\maketitle",
        convert(md),
        r"\end{document}",
    ])

    out_pdf = os.path.join(HERE, args.out)
    build = os.path.join(os.path.dirname(out_pdf), "_build")
    os.makedirs(build, exist_ok=True)
    tex_path = os.path.join(build, "paper2.tex")
    with open(tex_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tex)

    for run in (1, 2):                       # twice, for the page references
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", build, tex_path], cwd=HERE,
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tail = [l for l in proc.stdout.split("\n") if l.startswith("!")][:12]
            print(f"pdflatex failed on run {run}:")
            print("\n".join(tail) or proc.stdout[-2500:])
            print(f"\nthe .tex is at {os.path.relpath(tex_path, HERE)}")
            return 1

    shutil.copyfile(os.path.join(build, "paper2.pdf"), out_pdf)
    if not args.keep_tex:
        for ext in (".aux", ".log", ".out", ".toc"):
            p = os.path.join(build, "paper2" + ext)
            if os.path.exists(p):
                os.remove(p)
    print(f"wrote {os.path.relpath(out_pdf, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
