#!/bin/bash
# Build the OpenJaw paper PDF
# Usage: cd paper && bash build.sh

set -e
echo "=== Pass 1: pdflatex ==="
pdflatex -interaction=nonstopmode -halt-on-error main.tex
echo "=== Pass 2: bibtex ==="
bibtex main
echo "=== Pass 3: pdflatex ==="
pdflatex -interaction=nonstopmode -halt-on-error main.tex
echo "=== Pass 4: pdflatex ==="
pdflatex -interaction=nonstopmode -halt-on-error main.tex
echo "=== Done! ==="
echo "PDF: $(pwd)/main.pdf"
open main.pdf
