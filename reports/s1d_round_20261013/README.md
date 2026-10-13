# S1d complete round archive

See [final interpretation](../../experiments/S1d-round-20261013-results.md).

- `summary.json`: all final method and generation results.
- `training_results.csv` / `generation_results.csv`: compact tables.
- Original JSON reports contain histories, per-question correctness/predictions and hashes.
- Original logs, smoke results, audit certificates and preflight failures are preserved.
- `s1d_two_lines_v2.final_summary.json`: M1/M2/A1/A2 joint decision.
- `fp16_cache_manifest.json`: reader provenance, model weights remain on V100.
- `official_source/eval_other_utils.py`: official scorer source used for the audit.
- Runtime paths and hashes in original artifacts are preserved.
- Binary basis/alpha and fact gradient caches remain under `/share/latent/PRLM-C/runs/` on the shared V100 filesystem; their hashes and source paths appear in reports.
- Sixteen controlled test episodes remain sealed; no test predictions were generated.
