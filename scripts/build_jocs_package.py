"""Assemble the Journal of Computational Science upload bundle, and prove it builds.

Editorial Manager compiles uploaded LaTeX in ONE flat directory, while the repository
keeps the figure beside the script that draws it (``results/figures/``) and ``body.tex``
reaches it as ``../../results/figures/...``. Uploaded as-is, the manuscript would not
build. This script stages a flat copy, renames each figure ``Figure_<n>`` in order of
first inclusion (the guide's naming convention), rewrites the paths, and compiles the
staged copy from scratch before calling anything ready.

It writes ``docs/paper/submission/jocs_upload/``; each file's Editorial Manager item:

  neuroforge_cfd_elsevier.tex          Manuscript (LaTeX source file)
  preamble.tex abstract.tex body.tex   LaTeX source files
  refs.bib neuroforge_cfd_elsevier.bbl LaTeX source files
  Figure_1.pdf                         Figure (and a LaTeX source file)
  LaTeX_sources.zip                    all of the above, if EM offers a zip item
  neuroforge_cfd_elsevier.pdf          reference PDF; EM builds its own from the sources
  Highlights.docx                      Highlights
  Cover_letter.pdf                     Cover Letter (Cover_letter.txt to paste instead)
  Author_biographies.docx              Author Biography (vitae)

Not built here, deliberately: the Declaration of Interest file. The guide says "the
declarations tool should always be completed", so it must come from Elsevier's tool
("I have nothing to declare", saved as .docx), not from a look-alike made here.

Guarantees, each enforced rather than assumed:

* The staged sources are the committed ones with two changes only: full-line ``%``
  comments are dropped (they are working notes, not manuscript) and figure paths are
  rewritten. The staged build must reproduce a reference build of the unmodified
  sources label for label, number for number and page for page, with a byte-identical
  bibliography, which is what shows that neither change altered the paper.
* The staged build has zero errors, LaTeX/package/font warnings, overfull and underfull
  boxes, undefined references, duplicate PDF destinations and BibTeX warnings.
* The cover letter is everything in ``cover_letter.md`` above its ``---`` rule. The
  script refuses to build it unless the "Notes to self" heading is below that rule, and
  checks that the letter names the manuscript's exact title.
* No upload file contains a string in FORBIDDEN (private notes, the private address).
* Highlights: 3-5 bullets, each at most 85 characters including spaces.
* Vitae: at most 100 words each; unfilled ``[[FILL: ...]]`` markers are reported as
  NOT READY rather than silently shipped.

Word files are written as minimal OOXML with fixed timestamps, so a rebuild of
unchanged inputs is byte-identical and needs no third-party package.

Usage
-----
    python scripts/build_jocs_package.py
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from xml.sax.saxutils import escape as xml_escape

TEX_ROOT = os.path.join("docs", "paper")
SUBMISSION = os.path.join(TEX_ROOT, "submission")
MAIN = "neuroforge_cfd_elsevier"
SOURCES = [f"{MAIN}.tex", "preamble.tex", "abstract.tex", "body.tex", "refs.bib"]
EXPECTED_INPUTS = {"preamble", "abstract", "body"}
HIGHLIGHT_MAX_CHARS = 85
VITAE_MAX_WORDS = 100
FORBIDDEN = [r"Notes to self", r"kasraghanavati", r"icloud\.com", r"Kasra \(Ali\)",
             r"desk[- ]?reject", r"CMAME", r"Computers & Fluids", r"\bC&F\b"]
BUILD_JUNK = (".aux", ".log", ".out", ".blg", ".toc", ".lof", ".lot", ".fls",
              ".fdb_latexmk", ".synctex.gz", ".spl")
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
INCLUDE_RE = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]*)(\})")
LABEL_RE = re.compile(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}")


# --------------------------------------------------------------------------- LaTeX

def strip_comment_lines(text: str) -> str:
    """Drop lines that are entirely a comment. TeX discards such a line together with
    its end-of-line, so removing it cannot change the token stream."""
    return "".join(l for l in text.splitlines(keepends=True)
                   if not re.match(r"[ \t]*%", l))


def compile_tex(d: str, main: str, bibtex: bool) -> dict:
    for ext in BUILD_JUNK + (".bbl", ".pdf"):
        p = os.path.join(d, main + ext)
        if os.path.exists(p):
            os.remove(p)
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", main + ".tex"]
    steps = [latex] + ([["bibtex", main]] if bibtex else []) + [latex, latex]
    for s in steps:
        r = subprocess.run(s, cwd=d, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if r.returncode != 0:
            tail = "\n".join(r.stdout.strip().split("\n")[-25:])
            raise SystemExit(f"build failed in {d}: {' '.join(s)}\n{tail}")
    log = open(os.path.join(d, main + ".log"), encoding="utf-8", errors="replace").read()
    blg = (open(os.path.join(d, main + ".blg"), encoding="utf-8", errors="replace").read()
           if bibtex else "")
    pages = re.search(r"Output written on .*?\((\d+) pages?", log)
    return {
        "pages": int(pages.group(1)) if pages else -1,
        "errors": len(re.findall(r"^! ", log, re.M)),
        "overfull": len(re.findall(r"^Overfull", log, re.M)),
        "underfull": len(re.findall(r"^Underfull", log, re.M)),
        "latex warnings": len(re.findall(r"LaTeX Warning", log)),
        "package warnings": len(re.findall(r"Package \S+ Warning", log)),
        "font warnings": len(re.findall(r"Font Warning", log)),
        "undefined": len(re.findall(r"undefined", log, re.I)),
        "duplicate destinations": len(re.findall(r"destination with the same identifier",
                                                 log)),
        "bibtex warnings": len(re.findall(r"^Warning--", blg, re.M)),
    }


def require_clean(tag: str, res: dict) -> None:
    bad = {k: v for k, v in res.items() if k != "pages" and v}
    print(f"  {tag}: {res['pages']} pages, "
          + ("clean" if not bad else "PROBLEMS " + str(bad)))
    if bad:
        raise SystemExit(f"{tag} is not clean: {bad}")


def labels(aux_path: str) -> dict:
    aux = open(aux_path, encoding="utf-8", errors="replace").read()
    return {m.group(1): (m.group(2), m.group(3)) for m in LABEL_RE.finditer(aux)}


def reference_build(root: str) -> tuple[dict, dict, str]:
    """Build the UNMODIFIED sources in a scratch tree that keeps the repository's
    relative layout, so ../../results/figures resolves exactly as it does in place."""
    tmp = tempfile.mkdtemp(prefix="jocs_ref_")
    try:
        tex = os.path.join(tmp, TEX_ROOT)
        os.makedirs(tex)
        for name in SOURCES:
            shutil.copy2(os.path.join(root, TEX_ROOT, name), os.path.join(tex, name))
        body = open(os.path.join(tex, "body.tex"), encoding="utf-8").read()
        for m in INCLUDE_RE.finditer(body):
            src = os.path.normpath(os.path.join(root, TEX_ROOT, m.group(2)))
            dst = os.path.normpath(os.path.join(tex, m.group(2)))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        res = compile_tex(tex, MAIN, bibtex=True)
        bbl = open(os.path.join(tex, MAIN + ".bbl"), encoding="utf-8").read()
        return res, labels(os.path.join(tex, MAIN + ".aux")), bbl
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stage_manuscript(root: str, out: str) -> dict:
    texts = {}
    for name in SOURCES:
        t = open(os.path.join(root, TEX_ROOT, name), encoding="utf-8").read()
        texts[name] = strip_comment_lines(t) if name.endswith(".tex") else t

    inputs = set()
    for name, t in texts.items():
        if name.endswith(".tex"):
            inputs |= set(re.findall(r"\\(?:input|include)\{([^}]*)\}", t))
    if inputs != EXPECTED_INPUTS:
        raise SystemExit(f"unexpected \\input set {sorted(inputs)}; update SOURCES")
    for name in SOURCES:
        if name != "body.tex" and INCLUDE_RE.search(texts[name]):
            raise SystemExit(f"{name} includes a graphic; only body.tex is handled")

    order: list[str] = []
    for m in INCLUDE_RE.finditer(texts["body.tex"]):
        if m.group(2) not in order:
            order.append(m.group(2))
    figmap = {p: f"Figure_{i}{os.path.splitext(p)[1] or '.pdf'}"
              for i, p in enumerate(order, 1)}
    texts["body.tex"] = INCLUDE_RE.sub(
        lambda m: m.group(1) + figmap[m.group(2)] + m.group(3), texts["body.tex"])

    for name, t in texts.items():
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
    for src_rel, dst in figmap.items():
        src = os.path.normpath(os.path.join(root, TEX_ROOT, src_rel))
        if not os.path.isfile(src):
            raise SystemExit(f"figure referenced but missing: {src}")
        shutil.copy2(src, os.path.join(out, dst))
    return figmap


# ------------------------------------------------------------------ markdown sources

def paragraphs(md: str) -> list[list[str]]:
    """Blank-line separated blocks, each a list of its raw lines."""
    blocks, cur = [], []
    for line in md.split("\n"):
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def curly_quotes(s: str) -> str:
    out, opening = [], True
    for ch in s:
        if ch == '"':
            out.append("\u201c" if opening else "\u201d")
            opening = not opening
        else:
            out.append(ch)
    if not opening:
        raise SystemExit(f"unbalanced double quote in: {s[:80]}")
    return "".join(out)


def runs(s: str) -> list[tuple[str, bool, bool]]:
    """Split markdown emphasis into (text, bold, italic) runs."""
    out, buf, bold, ital, i = [], [], False, False, 0
    while i < len(s):
        if s.startswith("**", i) or s[i] == "*":
            if buf:
                out.append(("".join(buf), bold, ital))
                buf = []
            if s.startswith("**", i):
                bold, i = not bold, i + 2
            else:
                ital, i = not ital, i + 1
            continue
        buf.append(s[i])
        i += 1
    if buf:
        out.append(("".join(buf), bold, ital))
    if bold or ital:
        raise SystemExit(f"unbalanced emphasis in: {s[:80]}")
    return out


TOKEN_RE = re.compile(r"(?P<url>https?://[^\s)]+)|(?P<date>\d{4}-\d{2}-\d{2})"
                      r"|(?P<base>\d+)\^(?P<exp>\d+)|(?P<num>\d+(?:\.\d+)?)x(?![A-Za-z0-9])")
TEX_CHARS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
             "^": r"\textasciicircum{}", "|": r"\textbar{}", "<": r"\textless{}",
             ">": r"\textgreater{}", "\u2014": "---", "\u2013": "--",
             "\u201c": "``", "\u201d": "''", "\u2019": "'"}


def tex_plain(s: str) -> str:
    out = []
    for ch in s:
        if ch in TEX_CHARS:
            out.append(TEX_CHARS[ch])
        elif ord(ch) > 127:
            raise SystemExit(f"no LaTeX mapping for {ch!r} in: {s[:80]}")
        else:
            out.append(ch)
    return "".join(out)


def tex_text(s: str) -> str:
    out, last = [], 0
    for m in TOKEN_RE.finditer(s):
        out.append(tex_plain(s[last:m.start()]))
        if m.group("url"):
            out.append(r"\url{" + m.group("url") + "}")
        elif m.group("date"):
            out.append(r"\mbox{" + m.group("date") + "}")
        elif m.group("base"):
            out.append(f"${m.group('base')}^{{{m.group('exp')}}}$")
        else:
            out.append(m.group("num") + r"$\times$")
        last = m.end()
    out.append(tex_plain(s[last:]))
    return "".join(out)


def unicode_text(s: str) -> str:
    """The same substitutions for Word and plain text: 8.4x -> 8.4×, 128^2 -> 128²."""
    s = re.sub(r"(\d+)\^2\b", "\\1\u00b2", s)
    return re.sub(r"(\d+(?:\.\d+)?)x(?![A-Za-z0-9])", "\\1\u00d7", s)


def tex_runs(rs: list[tuple[str, bool, bool]]) -> str:
    out = []
    for text, b, i in rs:
        t = tex_text(text)
        if i:
            t = r"\emph{" + t + "}"
        if b:
            t = r"\textbf{" + t + "}"
        out.append(t)
    return "".join(out)


# ----------------------------------------------------------------------- Word files

def docx_bytes_write(path: str, paras: list[list[tuple[str, bool, bool]]]) -> None:
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = []
    for rs in paras:
        rr = []
        for text, b, i in rs:
            rpr = "<w:rPr>" + ("<w:b/>" if b else "") + ("<w:i/>" if i else "") + "</w:rPr>"
            parts = text.split("\n")
            inner = "<w:br/>".join(f'<w:t xml:space="preserve">{xml_escape(p)}</w:t>'
                                   for p in parts)
            rr.append(f"<w:r>{rpr}{inner}</w:r>")
        body.append("<w:p>" + "".join(rr) + "</w:p>")
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f"<w:document {ns}><w:body>" + "".join(body) +
                '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
                'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
                "</w:body></w:document>")
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              f"<w:styles {ns}><w:docDefaults><w:rPrDefault><w:rPr>"
              '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
              'w:cs="Times New Roman" w:eastAsia="Times New Roman"/>'
              '<w:sz w:val="24"/><w:szCs w:val="24"/><w:lang w:val="en-GB"/>'
              "</w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>"
              '<w:spacing w:after="160" w:line="276" w:lineRule="auto"/>'
              "</w:pPr></w:pPrDefault></w:docDefaults>"
              '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
              '<w:name w:val="Normal"/><w:qFormat/></w:style></w:styles>')
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
        'relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            'relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.'
            'org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>')
    doc_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
                'relationships"><Relationship Id="rId1" Type="http://schemas.'
                'openxmlformats.org/officeDocument/2006/relationships/styles" '
                'Target="styles.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in (("[Content_Types].xml", content_types), ("_rels/.rels", rels),
                           ("word/document.xml", document),
                           ("word/_rels/document.xml.rels", doc_rels),
                           ("word/styles.xml", styles)):
            z.writestr(zipfile.ZipInfo(name, ZIP_DATE), data.encode("utf-8"),
                       zipfile.ZIP_DEFLATED)


# ---------------------------------------------------------------- the upload files

def build_letter(root: str, out: str, title: str) -> str:
    md = open(os.path.join(root, SUBMISSION, "cover_letter.md"), encoding="utf-8").read()
    lines = md.split("\n")
    try:
        rule = lines.index("---")
    except ValueError:
        raise SystemExit("cover_letter.md has no '---' rule separating the notes")
    letter, notes = "\n".join(lines[:rule]), "\n".join(lines[rule:])
    if "Notes to self" in letter or "Notes to self" not in notes:
        raise SystemExit("the 'Notes to self' heading must sit below the '---' rule")
    letter = "\n".join(l for l in letter.split("\n") if not l.startswith("# "))
    flat = " ".join(letter.split())
    if " ".join(title.split()) not in flat:
        raise SystemExit(f"the cover letter does not name the manuscript title: {title}")

    tex_blocks, docx_paras, txt_blocks, signature = [], [], [], False
    for block in paragraphs(letter):
        if block[0].startswith("- "):
            items, cur = [], []
            for l in block:
                if l.startswith("- "):
                    if cur:
                        items.append(" ".join(cur))
                    cur = [l[2:].strip()]
                else:
                    cur.append(l.strip())
            items.append(" ".join(cur))
            items = [curly_quotes(it) for it in items]
            tex_blocks.append("\\begin{itemize}\n" + "\n".join(
                "  \\item " + tex_runs(runs(it)) for it in items) + "\n\\end{itemize}")
            for it in items:
                docx_paras.append([("\u2022 ", False, False)] +
                                  [(unicode_text(t), b, i) for t, b, i in runs(it)])
            txt_blocks.append("\n".join("- " + unicode_text(
                "".join(t for t, _, _ in runs(it))) for it in items))
            continue
        if signature:
            parts = [curly_quotes(l.strip()) for l in block]
            tex_blocks.append(" \\\\\n".join(tex_runs(runs(p)) for p in parts))
            docx_paras.append([(unicode_text("\n".join(parts)), False, False)])
            txt_blocks.append("\n".join(unicode_text(p) for p in parts))
            continue
        text = curly_quotes(" ".join(l.strip() for l in block))
        tex_blocks.append(tex_runs(runs(text)))
        docx_paras.append([(unicode_text(t), b, i) for t, b, i in runs(text)])
        txt_blocks.append(unicode_text("".join(t for t, _, _ in runs(text))))
        if text.strip() == "Sincerely,":
            signature = True

    tex = ("\\documentclass[11pt,a4paper]{article}\n"
           "\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n"
           "\\usepackage[margin=25mm]{geometry}\n\\usepackage{microtype}\n"
           "\\usepackage[hyphens]{url}\n\\usepackage{enumitem}\n"
           "\\setlength{\\parindent}{0pt}\n\\setlength{\\parskip}{0.6\\baselineskip}\n"
           "\\emergencystretch=2em\n"
           "\\setlist[itemize]{leftmargin=1.5em,itemsep=0.25\\baselineskip,topsep=0pt}\n"
           "\\begin{document}\n\n" + "\n\n".join(tex_blocks) + "\n\n\\end{document}\n")
    tmp = tempfile.mkdtemp(prefix="jocs_letter_")
    try:
        with open(os.path.join(tmp, "Cover_letter.tex"), "w", encoding="utf-8",
                  newline="\n") as fh:
            fh.write(tex)
        require_clean("cover letter", compile_tex(tmp, "Cover_letter", bibtex=False))
        shutil.copy2(os.path.join(tmp, "Cover_letter.pdf"),
                     os.path.join(out, "Cover_letter.pdf"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    txt = "\n\n".join(txt_blocks) + "\n"
    with open(os.path.join(out, "Cover_letter.txt"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(txt)
    return txt


def build_highlights(root: str, out: str) -> str:
    src = open(os.path.join(root, SUBMISSION, "highlights.txt"), encoding="utf-8").read()
    bullets = [l[2:].strip() for l in src.split("\n") if l.startswith("- ")]
    if not 3 <= len(bullets) <= 5:
        raise SystemExit(f"highlights: {len(bullets)} bullets, the guide allows 3-5")
    for b in bullets:
        if len(b) > HIGHLIGHT_MAX_CHARS:
            raise SystemExit(f"highlight over {HIGHLIGHT_MAX_CHARS} characters: {b}")
    docx_bytes_write(os.path.join(out, "Highlights.docx"),
                     [[("Highlights", True, False)]] +
                     [[("\u2022 " + b, False, False)] for b in bullets])
    print(f"  highlights: {len(bullets)} bullets, longest "
          f"{max(len(b) for b in bullets)} characters")
    return "\n".join(bullets)


def build_vitae(root: str, out: str) -> tuple[str, list[str]]:
    md = open(os.path.join(root, SUBMISSION, "vitae.md"), encoding="utf-8").read()
    sections = re.split(r"^## (.+)$", md, flags=re.M)[1:]
    if len(sections) != 4:
        raise SystemExit("vitae.md must hold exactly two '## Name' sections")
    paras, problems, text = [], [], []
    for name, body in zip(sections[0::2], sections[1::2]):
        bio = " ".join(body.split())
        words = len(bio.split())
        fills = re.findall(r"\[\[FILL:[^\]]*\]\]", bio)
        print(f"  vita {name.strip()}: {words} words"
              + (f", {len(fills)} unfilled marker(s)" if fills else ""))
        if words > VITAE_MAX_WORDS:
            raise SystemExit(f"vita for {name} is {words} words; the limit is 100")
        if fills:
            problems.append(f"vita for {name.strip()}: {len(fills)} [[FILL]] marker(s)")
        paras += [[(name.strip(), True, False)], [(bio, False, False)]]
        text.append(bio)
    docx_bytes_write(os.path.join(out, "Author_biographies.docx"), paras)
    return "\n".join(text), problems


def zip_sources(out: str, names: list[str]) -> str:
    path = os.path.join(out, "LaTeX_sources.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            with open(os.path.join(out, n), "rb") as fh:
                z.writestr(zipfile.ZipInfo(n, ZIP_DATE), fh.read(), zipfile.ZIP_DEFLATED)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=os.path.join(SUBMISSION, "jocs_upload"))
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    out = os.path.join(root, args.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    main_tex = open(os.path.join(root, TEX_ROOT, f"{MAIN}.tex"), encoding="utf-8").read()
    m = re.search(r"\\title\{([^}]*)\}", strip_comment_lines(main_tex))
    title = " ".join(m.group(1).replace("\\\\", " ").split())
    print(f"title: {title}")

    print("reference build of the unmodified sources")
    ref_res, ref_labels, ref_bbl = reference_build(root)
    require_clean("reference build", ref_res)

    print("staging the flat upload copy")
    figmap = stage_manuscript(root, out)
    for src, dst in figmap.items():
        print(f"  {dst} <- {os.path.normpath(os.path.join(TEX_ROOT, src))}")
    res = compile_tex(out, MAIN, bibtex=True)
    require_clean("staged build", res)
    got = labels(os.path.join(out, MAIN + ".aux"))
    if got != ref_labels or res["pages"] != ref_res["pages"]:
        diff = sorted(k for k in set(got) | set(ref_labels) if got.get(k) != ref_labels.get(k))
        raise SystemExit(f"staged build differs from the reference: {diff[:10]}")
    if open(os.path.join(out, MAIN + ".bbl"), encoding="utf-8").read() != ref_bbl:
        raise SystemExit("staged bibliography differs from the reference build's")
    print(f"  identical to the reference: {len(got)} labels, {res['pages']} pages, "
          "bibliography byte-identical")
    for n in os.listdir(out):
        if n.endswith(BUILD_JUNK):
            os.remove(os.path.join(out, n))
    tex_files = SOURCES + [f"{MAIN}.bbl"] + sorted(figmap.values())
    zip_sources(out, tex_files)

    print("building the other upload files")
    letter = build_letter(root, out, title)
    highlights = build_highlights(root, out)
    vitae, problems = build_vitae(root, out)

    scanned = {n: open(os.path.join(out, n), encoding="utf-8").read()
               for n in SOURCES}
    scanned.update({"Cover_letter": letter, "Highlights": highlights,
                    "Author_biographies": vitae})
    hits = [(n, p) for n, t in scanned.items() for p in FORBIDDEN if re.search(p, t)]
    if hits:
        raise SystemExit(f"forbidden strings in upload files: {hits}")
    print(f"  no forbidden strings in {len(scanned)} upload texts")

    print(f"\nwrote {os.path.relpath(out, root)}:")
    for n in sorted(os.listdir(out)):
        print(f"  {n:36s} {os.path.getsize(os.path.join(out, n)) / 1024:8.1f} KB")
    if problems:
        print("\nNOT READY -- the bundle builds, but these need the author:")
        for p in problems:
            print(f"  - {p}")
        return 2
    print("\nREADY: every upload file is built and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
