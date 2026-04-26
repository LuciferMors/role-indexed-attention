#!/bin/bash
# Build a fully-anonymized NeurIPS submission package.
#
# Output: paper/neurips_submission/  with:
#   main.pdf              (anonymized paper)
#   supplementary.zip     (code + extra results)

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="$ROOT/paper/neurips_submission"
mkdir -p "$OUT"

# 1. Build anonymized paper from main_neurips.tex.
cd "$ROOT/paper"
# Wipe author info entirely (already anonymous, but be defensive).
python3 - <<'PY'
src = open('main_neurips.tex').read()
# Strip any potential author identifiers.
import re
src = re.sub(r'lucifermorsbio@gmail\.com', '[anonymized]', src)
src = re.sub(r'rishivhare', '[anonymized]', src, flags=re.IGNORECASE)
src = re.sub(r'Rishi Vhare', 'Anonymous Author', src)
src = src.replace('github.com/rishivhare', '[anonymized]')
open('main_neurips_anon.tex', 'w').write(src)
PY

pdflatex -interaction=nonstopmode -jobname=main_neurips_anon main_neurips_anon.tex > /dev/null 2>&1 || true
bibtex main_neurips_anon > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode -jobname=main_neurips_anon main_neurips_anon.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode -jobname=main_neurips_anon main_neurips_anon.tex > /dev/null 2>&1 || true

# Strip PDF metadata (author/title may include name).
exiftool -overwrite_original \
  -Title="" -Author="" -Subject="" -Keywords="" -Creator="" -Producer="" \
  main_neurips_anon.pdf 2>/dev/null || \
  echo "WARN: install exiftool (brew install exiftool) to scrub PDF metadata"

cp main_neurips_anon.pdf "$OUT/main.pdf"
echo "Anonymized paper -> $OUT/main.pdf"

# 2. Build supplementary zip (code + appendix + results).
cd "$ROOT"
zip -r "$OUT/supplementary.zip" \
  ava/ \
  experiments/ \
  tests/ \
  Makefile \
  results/FINAL_RESULTS.md \
  results/gpu/ \
  -x "**/__pycache__/*" -x "**/.DS_Store" -x "**/*.pyc"

echo ""
echo "Submission package ready in $OUT:"
ls -la "$OUT"
echo ""
echo "Now manually verify in main.pdf that NO author / affiliation / GitHub URL appears."
echo "Use:  open $OUT/main.pdf"
