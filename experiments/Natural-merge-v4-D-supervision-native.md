# D supervision repair and native source skip (v4)

This comparison separates supervision repair from one architectural change. It
uses the original natural-data training contexts and the existing development
E1 plan. It does not start formal E1/E2/E3 or select checkpoints using results.

| Variant | Change from v3 D |
| --- | --- |
| Dsup | Answer boundary repair, independent true EOS CE, full original QA coverage |
| Dnative | Identical supervision, plus one full-width source skip using existing routing |

Both variants start fresh with seed 0 and follow the same sealed 64-update
single-merge/recurrent schedule. All 1,402 original training QAs enter at least
one training event. Losses are balanced by source context, so a context with more
questions does not automatically receive greater weight. This changes the
amount of supervision per update relative to v3; Dsup versus old D measures the
combined supervision repair, not an isolated EOS ablation.

Answer content is encoded as one leading ASCII space followed by the unchanged
original answer. The tokenizer's actual EOS token, id 128001, is appended.
Recovery CE, eligible Full KL and protection hinge operate on answer content
only. EOS CE is computed independently, with weight 0.25. There is no Context or
newline suppression penalty. Full teacher eligibility excludes EOS.

Dnative reuses Top-8 routing, slot gates, layer groups, and the 5% global RMS
budget. It adds the routed full-width source average minus the carrier, scaled
by one zero-initialized signed scalar. At initialization its output equals
Dsup's output. The low-rank and native branches are jointly gated and capped;
no slots are added. This is a routed mixture, not lossless copying. There is no
head-slot mask, forced balancing, or explicit memory partition.

## Execution

On `v100-2`, from `/share/latent/PRLM-C`:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /share/latent/mllm_env/bin/python tools/run_natural_merge_v4.py \
  --train-plan data/natural_merge_v4/train.json \
  --dev-plan data/natural_merge_v1/E1-development.json \
  --model-path /dev/shm/s1d_reader_fp16 \
  --devices 7,9 --variants Dsup,Dnative \
  --out-dir runs/natural_merge_v4_dd_20261015_0108
```

The launcher runs one smoke update, then fresh 64-update training, a fixed
training-window fitting diagnosis, and frozen development E1 generation for
each variant. It records commands, PID, physical GPU, exit codes and logs in
`launcher.json`. It refuses an existing output directory and never terminates
unrelated jobs. Training-fit results are explicitly excluded from generalization
claims. Each development result saves original QA predictions and verification.

Training plan content SHA-256:
`40072a9a3edb2a0b325b2f4d50f920c1123763e954f05c8b27dac8d4d7944d8c`.

## Checks and interpretation

Six local CPU tests pass, covering immutable QA/EOS encoding, context-balanced
losses, independent EOS/teacher terms, native-path initialization/gradients/mask
and budget, reader-gradient bridging, and checkpoint architecture validation.
The server-generated plan validates 64 events and all 1,402 unique QA ids.

Compare Dsup with v3 D to assess the supervision repair; compare Dnative with
Dsup to isolate the source skip. Report author containment together with
normalized EM/F1, compression-loss recovery, discarded-source gains/harms,
generation termination and per-discarded-block routing mass. A better training
fit alone does not establish held-out performance. No new result is claimed
until the jobs finish.
