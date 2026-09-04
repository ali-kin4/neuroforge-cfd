#!/usr/bin/env bash
# The fixed-point control (scripts/perturbation_probe.py), one process per case.
#
# Five cases x four arms plus a cold baseline. `solve_cgrid(reuse=True)` reads
# back anything already finished, so re-running after an interruption is free.
#
#     bash scripts/run_perturbation.sh
#
set -u

PY=.venv/Scripts/python.exe
CASES=("naca0012@0" "naca0012@4" "naca0015@6" "naca2412@2" "naca2415@5")
mkdir -p logs

for spec in "${CASES[@]}"; do
  tag=$(echo "$spec" | tr '@' '_')
  echo "launch $spec"
  "$PY" scripts/perturbation_probe.py --only "$spec" \
      > "logs/perturb_${tag}.log" 2>&1 &
done
wait
echo "all five cases done"
