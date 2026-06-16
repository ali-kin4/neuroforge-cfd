# NeuroForge — full research run summary

- generated: 2026-06-15T10:33:13.328095+00:00
- git commit: `d2c5556`
- device: NVIDIA GeForce RTX 4070 Ti (12.9 GB)
- torch 2.6.0+cu124, CUDA 12.4
- preset: `full`  (airfrans/full, 800/200, seeds [0, 1, 2])

## Stage timings (this invocation; skipped stages omitted)

| stage | wall-clock |
|---|---|
| in-distribution ablation (Table 1) | 5.87 h |
| out-of-distribution ablation (Table 3) | 10.96 h |
| **total (this run)** | **16.83 h** |

## Artifacts

- `in_dist/ablation.md` / `.csv` — Table 1 (arms x metrics, mean±std)
- `ood/ablation_ood.md` / `.csv` — Table 3 (in-dist -> OOD gap)
- `MANIFEST.json` — environment + exact config for reproducibility

Re-running this script skips completed stages; pass `--force` to recompute.
