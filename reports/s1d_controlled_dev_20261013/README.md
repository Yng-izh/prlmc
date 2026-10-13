# S1d controlled development archive — 2026-10-13

[Chinese result report](../../experiments/S1d-controlled-development-20261013-results.md) · [Native-policy paper audit](../../experiments/S1d-native-policy-paper-audit-20261013.md).

- `summary.json`: consolidated results and limitations; `generation_results.csv` / `c13_results.csv`: episode tables.
- `runs/`: original shared E6d (8 complete), native policy (8 complete), and C13 (4 complete / 4 rank failures) JSON, raw generation predictions, histories, logs, launcher/fleet records and C13 smoke evidence.
- Binary C13 alpha/basis files and the shared E6d checkpoint remain on v100-2. Their original paths, full hashes and byte sizes are archived. The identical controlled input plan is already tracked at `runs/sic_paper_controlled_plan.json`.
- `source_snapshot/memoryllm_modeling/`: server reader implementation/configuration and original pre-patch backup, saved as source provenance snapshots; these are not installed or modified in the live source tree.
- `REMOTE_ARTIFACT_MANIFEST.json`: original server paths, hashes, byte sizes and archive inclusion status. All `.pt` files remain on `/share/latent/PRLM-C/runs/`; their full hashes/paths are retained, including the frozen alpha checkpoint with its embedded inference basis.
- `SHA256SUMS.json`: hashes of all archived files (excluding the checksum file itself).
- `validation.json`: independent rescoring, downloaded-file hash verification, frozen/report/budget checks and CPU test record. No GPU reruns were needed for archiving.

Original absolute/runtime paths inside reports are intentionally preserved. The new top-level summary is editorial and does not replace the original reports. Existing older-round results remain in `reports/s1d_round_20261013/`.

C13 uses original C gamma_max=0.1; shared E6d evaluation uses its training gamma_max=0.05. They are distinct effective configurations despite the identical carrier checkpoint hash. Partial C13 results are not an eight-episode benchmark.

All evaluation is controlled development only. No sealed-test predictions were generated. The cold-start versus checkpoint-memory comparison and any new shared-module training were **not started**.
