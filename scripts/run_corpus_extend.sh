#!/usr/bin/env bash
# Add the two decomposition arms (`oracle_bl`, `or_proj_coarse`) to the thirteen
# corpus cases. Every other arm is already on disk and `solve_cgrid(reuse=True)`
# reads it back instead of re-solving, so this costs 26 solves, not 65.
#
# One process per case, five at a time: simpleFoam is serial here and the
# threading cap in neuroforge/__init__.py holds each process to one core.
#
#     bash scripts/run_corpus_extend.sh
#
set -u

PY=.venv/Scripts/python.exe
CASES=(
  "naca0012@8" "naca0012@10" "naca0012@12" "naca2412@8" "naca2412@10"
  "naca0018@4" "naca0018@8" "naca4415@4" "naca2415@8" "naca0015@2"
  "naca2415@2" "naca4415@2" "naca0015@4"
)
mkdir -p logs

running=0
for spec in "${CASES[@]}"; do
  tag=$(echo "$spec" | tr '@' '_')
  echo "launch $spec"
  "$PY" scripts/corpus_probe.py --only "$spec" \
      > "logs/corpusext_${tag}.log" 2>&1 &
  running=$((running + 1))
  if [ "$running" -ge 5 ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait
echo "all thirteen cases done"
