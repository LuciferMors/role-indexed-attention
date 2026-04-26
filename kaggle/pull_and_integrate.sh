#!/bin/bash
# Pull kernel output, integrate into paper.
set -e

export KAGGLE_USERNAME=rishivhare
export KAGGLE_API_TOKEN=KGAT_f4d9ad6ecc33ad09a1cc8d39b2f2dec9
KAGGLE=~/Library/Python/3.12/bin/kaggle
ROOT=/Users/rishi/Desktop/y/avacchedaka
SLUG=rishivhare/avacchedaka-attention-battery-t4x2

mkdir -p "$ROOT/results/gpu"
cd "$ROOT/results/gpu"
echo "Pulling kernel output..."
$KAGGLE kernels output "$SLUG" -p . 2>&1 | tail -10

echo ""
echo "Files retrieved:"
ls -la

if [ -f results.jsonl ]; then
  echo ""
  echo "Result count: $(wc -l < results.jsonl)"
  echo "First 3 lines:"
  head -3 results.jsonl
fi

if [ -f avacchedaka-attention-battery-t4x2.log ]; then
  echo ""
  echo "Last 30 log lines:"
  tail -30 avacchedaka-attention-battery-t4x2.log
fi
