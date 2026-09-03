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

# elsarticle wants a frontmatter block, not \maketitle: authors carry affiliation
# labels, the corresponding author is marked with \corref, and the abstract and
# keywords live inside the block rather than after it.
AUTHORS = r"""\author[urmia]{Ali Jabbary\corref{cor1}}
\ead{st_a.jabbary@urmia.ac.ir}
\affiliation[urmia]{organization={Department of Mechanical Engineering,
  Urmia University}, city={Urmia}, country={Iran}}

\author[gre]{Kasra Ghanavati}
\affiliation[gre]{organization={School of Computing and Mathematical Sciences,
  University of Greenwich}, city={London}, country={United Kingdom}}

\cortext[cor1]{Corresponding author.}"""

PREAMBLE = r"""% `number` (not authoryear): the bibliography is a hand-written
% thebibliography, and natbib's author-year mode rejects it outright.
\documentclass[preprint,11pt,number]{elsarticle}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage[skip=4pt]{caption}
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


# Transliteration used inside code spans only. Keys are the non-ASCII this
# paper actually uses; anything unlisted would still reach pdflatex and stop it,
# which is the loud failure we want rather than a silently mangled symbol.
CODE_ASCII = str.maketrans({
    "⁺": "+", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6",
    "₇": "7", "₈": "8", "₉": "9", "₋": "-",
    "τ": "tau", "ν": "nu", "κ": "kappa", "ω": "omega", "δ": "delta",
    "α": "alpha", "×": "x", "·": ".", "−": "-", "≈": "~", "≤": "<=",
    "≥": ">=", "±": "+/-", "—": "--", "–": "-", "°": "deg",
})


def restore_code(text: str, store: list[str]) -> str:
    def sub(m):
        body = store[int(m.group(1))]
        # A URL in a code span must be \url{}: \texttt{} will not break it, and
        # the repository link ran off the right margin in the first elsarticle
        # build.
        if body.startswith(("http://", "https://", "www.")):
            return r"\url{" + body + "}"
        # Inside monospace the maths-mode substitutions of `unicode_to_tex` are
        # both ugly and wrong -- `\texttt{$y^{+}$}` is not what a reader wants
        # from `y+` -- and a stray superscript byte stops pdflatex outright. So
        # code spans get an ASCII transliteration instead, which is what a
        # reader of monospace expects anyway.
        body = body.translate(CODE_ASCII)
        for a, b in (("\\", r"\textbackslash{}"), ("{", r"\{"), ("}", r"\}"),
                     ("_", r"\_"), ("^", r"\^{}"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#"), ("~", r"\textasciitilde{}")):
            body = body.replace(a, b)
        return r"\texttt{" + body + "}"
    return re.sub("\x00(\\d+)\x00", sub, text)


# The exact non-ASCII inventory of the draft, checked with a character census
# rather than guessed. Superscript runs are handled before the singles so that
# "10^-5" written as digits plus U+207B collapses into one math group.
SUPERS = {"\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2070": "0",
          "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
          "\u2078": "8", "\u2079": "9", "\u207b": "-", "\u207a": "+"}
SUBS = {"\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3",
        "\u2084": "4", "\u2085": "5", "\u2086": "6", "\u2087": "7",
        "\u2088": "8", "\u2089": "9", "\u208b": "-"}
SINGLES = [
    ("\u2014", "---"), ("\u2013", "--"), ("\u00a7", r"\S"),
    ("\u2212", "$-$"), ("\u00b0", r"\textdegree{}"), ("\u00d7", r"$\times$"),
    ("\u2192", r"$\rightarrow$"), ("\u2264", r"$\le$"), ("\u2265", r"$\ge$"),
    ("\u2248", r"$\approx$"), ("\u00b7", r"$\cdot$"), ("\u00b1", r"$\pm$"),
    ("\u26a0", r"$\rightarrow$"),
    # Greek from the wall-law expressions. `elsarticle` is not a maths package,
    # so these must become maths mode or pdflatex stops on the byte.
    ("\u03c4", r"$\tau$"), ("\u03bd", r"$\nu$"), ("\u03ba", r"$\kappa$"),
    ("\u03c9", r"$\omega$"), ("\u03b4", r"$\delta$"), ("\u03b1", r"$\alpha$"),
]


def unicode_to_tex(text: str) -> str:
    def sup(m):
        return "$^{" + "".join(SUPERS[c] for c in m.group(0)) + "}$"

    def sub(m):
        return "$_{" + "".join(SUBS[c] for c in m.group(0)) + "}$"
    text = re.sub("[" + "".join(SUPERS) + "]+", sup, text)
    text = re.sub("[" + "".join(SUBS) + "]+", sub, text)
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


def split_sections(md: str) -> dict:
    """`## Name` blocks, for the pieces that are front matter rather than body."""
    out, name, buf = {}, None, []
    for line in md.split("\n"):
        if line.startswith("## "):
            if name:
                out[name] = "\n".join(buf).strip()
            name, buf = line[3:].strip(), []
        elif name:
            buf.append(line)
    if name:
        out[name] = "\n".join(buf).strip()
    return out


def split_numbered(text: str) -> list[str]:
    """A `1. ...` list, with wrapped continuation lines rejoined."""
    items: list[str] = []
    for line in text.split("\n"):
        if re.match(r"^\s*\d+\.\s", line):
            items.append(line.strip())
        elif items and line.strip():
            items[-1] += " " + line.strip()
    return items


def convert_body(md: str) -> str:
    """Paragraph-only conversion, for the abstract."""
    return "\n\n".join(inline(p.replace("\n", " ").strip())
                       for p in md.split("\n\n") if p.strip())


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
            # A blockquote carrying ![](path) is a figure with its caption: emit
            # a real float so the caption travels with the image and the numbers
            # in the text ("Figure 1") match what a reader sees.
            images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
            caption = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text).strip()
            if images:
                out.append(r"\begin{figure}[htbp]\centering")
                for path in images:
                    if os.path.isfile(os.path.join(HERE, path)):
                        out.append(r"\includegraphics[width=\textwidth]{"
                                   + path + "}")
                out.append(r"\caption*{\small " + inline(caption) + "}")
                out.append(r"\end{figure}")
            else:
                out.append(r"\begin{shaded}\noindent " + inline(text)
                           + r"\end{shaded}")
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

    sections = split_sections(md)
    for required in ("Abstract", "Keywords", "Highlights", "References"):
        if required not in sections:
            print(f"the draft has no '## {required}' section")
            return 1

    # Escape before inserting LaTeX, or the line break is escaped with the prose.
    raw = md.split("\n", 1)[0].lstrip("# ").strip()
    head, sep, tail = raw.partition(":")
    title = unicode_to_tex(escape(head.strip()))
    if sep:
        title += ":" + r"\\[0.35em] \large " + unicode_to_tex(escape(tail.strip()))

    keywords = r" \sep ".join(
        inline(k.strip()) for k in
        re.split(r"[;\n]", sections["Keywords"]) if k.strip())

    frontmatter = "\n".join([
        r"\begin{frontmatter}",
        r"\title{" + title + "}",
        AUTHORS,
        r"\begin{abstract}",
        convert_body(sections["Abstract"]),
        r"\end{abstract}",
        r"\begin{keyword}",
        keywords,
        r"\end{keyword}",
        r"\end{frontmatter}",
    ])

    # Elsevier wants Highlights as a separate file at submission; they are also
    # shown here so the PDF is self-contained for a reader.
    bullets = [b.strip() for b in re.findall(r"^- (.+)$",
                                             sections["Highlights"], re.M)]
    over = [b for b in bullets if len(b) > 85]
    if over:
        print("highlights over the 85-character limit:")
        for b in over:
            print(f"  {len(b)}  {b}")
        return 1
    hl_path = os.path.join(os.path.dirname(os.path.join(HERE, args.out)),
                           "highlights.txt")
    with open(hl_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join("- " + b for b in bullets) + "\n")

    highlights_tex = "\n".join(
        [r"\section*{Highlights}", r"\begin{itemize}"]
        + [r"\item " + inline(b) for b in bullets]
        + [r"\end{itemize}"])

    bib = "\n".join(
        [r"\begin{thebibliography}{99}"]
        + [r"\bibitem{r%d} " % k + inline(re.sub(r"^\d+\.\s*", "", e))
           for k, e in enumerate(split_numbered(sections["References"]), 1)]
        + [r"\end{thebibliography}"])

    body_md = md[md.index("## 1. Introduction"):md.index("## References")]
    tex = "\n".join([
        PREAMBLE,
        r"\begin{document}",
        frontmatter,
        highlights_tex,
        convert(body_md),
        bib,
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
            # The log echoes the offending source bytes, so it is routinely
            # non-ASCII -- and printing it under a redirected cp1252 stdout
            # raises, hiding the real error behind a UnicodeEncodeError.
            message = "\n".join(tail) or proc.stdout[-2500:]
            print(message.encode("ascii", "backslashreplace").decode("ascii"))
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
