# results/

Experiment outputs live here and **are committed to the repo** (this replaced
the old Google-Drive backup). The workflow is: run on your PC, get results
locally in this folder, then push them back.

What lands here (all small, text/figures — safe for git):

| File | Written by | Meaning |
|---|---|---|
| `full_research/in_dist/ablation.{md,csv}` | `scripts/run_full_research.py` | Table 1 — in-distribution ablation |
| `full_research/ood/ablation_ood.{md,csv}` | `scripts/run_full_research.py` | Table 3 — OOD ablation |
| `full_research/MANIFEST.json` | `scripts/run_full_research.py` | env + exact config (reproducibility) |
| `full_research/SUMMARY.md` | `scripts/run_full_research.py` | per-stage wall-clock timings |
| `ablation.{md,csv}`, `ablation_ood.{md,csv}`, `calibration.md`, `REPORT.md` | notebooks / `benchmarks/ablation.py` | evidence-pack tables |

What does **not** go here (kept out by `.gitignore`): trained checkpoints
(`checkpoints/*.pt`), the AirfRANS download + rasterised cache (`data/`), and any
`*.pt`/`*.ckpt`/`*.vtk` binaries — these are large and regenerated, not results.

## Push results back to the repo

```bash
python scripts/push_results.py            # stages results/, commits, pushes
# or manually:
git add results/ && git commit -m "results: <run>" && git push
```
