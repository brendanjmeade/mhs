#!/usr/bin/env python
"""Build the mhs figures: TikZ -> PDF (paper) and -> SVG (web).

    python figures/build.py                 # both figures, both variants
    python figures/build.py triangle        # just one
    python figures/build.py --png           # also rasterise, for looking at

Each figure is one `<name>.tex` holding a bare `tikzpicture`. A driver is
generated per variant, so the drawing is written once and the two renderings
differ only in `common.tex`'s palette and font block:

    web      sans text, the site's --accent / --fg / --fg-dim
    paper    Times text and math, black edges, single-hue fill

WHY tectonic AND mutool. tectonic is a single binary with no TeX installation;
`dvisvgm`, the usual PDF->SVG step, pulls in all of TeX Live, imagemagick and
perl through Homebrew, which would undo that. mupdf-tools has NO dependencies
and does both jobs -- `mutool convert` for the SVG and `mutool draw` for a PNG
to actually look at before shipping.

Outputs are committed. The Pages deploy does not run LaTeX and should not: the
same reasoning that keeps website/public/data/ in the repository.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
WEB = HERE.parent / "website" / "src" / "figures"
FIGURES = ("triangle", "mindlin")
MODES = ("web", "paper")


def run(cmd: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"\n$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}\n")
        raise SystemExit(f"{cmd[0]} failed with {r.returncode}")
    return r


def build(name: str, mode: str, png: bool) -> None:
    OUT.mkdir(exist_ok=True)
    stem = f"{name}-{mode}"
    driver = OUT / f"{stem}.tex"
    driver.write_text(
        f"\\def\\MODE{{{mode}}}\n"
        f"\\input{{{HERE / 'common.tex'}}}\n"
        f"\\begin{{document}}\n"
        f"\\input{{{HERE / (name + '.tex')}}}\n"
        f"\\end{{document}}\n")

    r = run(["tectonic", "-X", "compile", driver.name, "--outdir", "."], OUT)
    # A missing glyph is otherwise invisible: TeX reports it and renders a gap.
    for line in (r.stderr or "").splitlines():
        if "Missing character" in line or "not found" in line.lower():
            print(f"  !! {line.strip()}")
    driver.unlink()
    pdf = OUT / f"{stem}.pdf"

    if mode == "web":
        # mutool numbers its output per page; one page, so <stem>1.svg.
        run(["mutool", "convert", "-F", "svg", "-o", f"{stem}%d.svg",
             pdf.name], OUT)
        svg = OUT / f"{stem}1.svg"
        final = OUT / f"{stem}.svg"
        svg.replace(final)
        WEB.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, WEB / f"{name}.svg")
        size = final.stat().st_size
        print(f"  {name:9s} {mode:5s}  {size / 1024:6.1f} kB svg "
              f"-> website/src/figures/{name}.svg")
        pdf.unlink()                       # the web PDF is an intermediate
    else:
        print(f"  {name:9s} {mode:5s}  {pdf.stat().st_size / 1024:6.1f} kB pdf")

    if png:
        src = pdf if mode == "paper" else None
        if src is None:        # rebuild a PDF just to look at the web variant
            return
        run(["mutool", "draw", "-r", "200", "-o", f"{stem}.png", src.name], OUT)
        print(f"             {(OUT / (stem + '.png')).stat().st_size / 1024:6.1f} kB png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("figures", nargs="*", default=None,
                    help=f"which to build (default: {' '.join(FIGURES)})")
    ap.add_argument("--png", action="store_true",
                    help="also rasterise the paper variant, to look at it")
    a = ap.parse_args()
    names = a.figures or list(FIGURES)
    for n in names:
        if not (HERE / f"{n}.tex").exists():
            raise SystemExit(f"no such figure: {n}.tex")
        for m in MODES:
            build(n, m, a.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
