"""Compile in an isolated directory and publish the complete PDF atomically."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/"paper"
BUILD = ROOT/"tmp/latex-final"


def main():
    BUILD.mkdir(parents=True,exist_ok=True)
    # Old in-place LaTeX auxiliaries can shadow files in an output directory.
    for ext in ("aux","bbl","blg","fdb_latexmk","fls","log","out","toc"):
        (PAPER/f"main.{ext}").unlink(missing_ok=True)
        (BUILD/f"main.{ext}").unlink(missing_ok=True)
    shutil.copyfile(PAPER/"references.bib",BUILD/"references.bib")
    command = ["pdflatex","-interaction=nonstopmode","-halt-on-error",
               f"-output-directory={BUILD}","main.tex"]
    def run(cmd,cwd,name):
        result = subprocess.run(cmd,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (BUILD/name).write_bytes(result.stdout)
        if result.returncode:
            raise RuntimeError(f"Build failed; see {BUILD/name}")
    run(command,PAPER,"pass1.txt")
    run(["bibtex","main"],BUILD,"bibliography.txt")
    run(command,PAPER,"pass2.txt")
    run(command,PAPER,"pass3.txt")
    log = (BUILD/"main.log").read_text()
    for token in ("Overfull","undefined references","undefined citations","LaTeX Error"):
        if token in log:
            raise RuntimeError(f"PDF validation failed: {token}")
    source = BUILD/"main.pdf"
    data = source.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-1024:]:
        raise RuntimeError("Incomplete PDF")
    temporary = PAPER/"main.pdf.partial"
    temporary.write_bytes(data)
    temporary.replace(PAPER/"main.pdf")
    print(f"Complete PDF: {PAPER/'main.pdf'} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
