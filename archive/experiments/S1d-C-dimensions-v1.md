# C dimension ablation: 20D / 24D / 28D

本协议对应实验已完成；最终结果、判门与归档见 [本轮总报告](S1d-round-20261013-results.md)。
User authorized original C bases for all three variants. Freeze the weak
development window `(3,17510)`, cached damaged-fact gradient rows, original C
tail-4-token summed training gradients, AdamW lr 0.003 / weight decay 0.01,
displacement norm budget 1.0, seed 0, 60 steps, measurement every 10, QA cap
256 and TF-only evaluation. No protected projection or hinge objective.

Use the top 20, 24 or 28 singular vectors of the same original C gradient
matrix, comparing with existing C13 and C32. Reject insufficient rank.
Each run performs a one-step smoke and verification before 60-step training,
and verifies its final report and alpha checkpoint. Compare final step 60;
intermediate scores do not select a checkpoint. These are development ablations,
not sealed-test results or evidence of independent generalization.

Dedicated `s1d_dimension_basis.py`, `s1d_dimension_train.py` and
`verify_s1d_dimensions.py` keep active projected-C source files and cache
provenance unchanged. Carrier, row caches, fact cache, C reference and reader
manifest hashes were confirmed identical on v100-1 and v100-2 before launch.

Planned assignments:

| Variant | Host | GPU |
| --- | --- | ---: |
| C20 | v100-1 | 1 |
| C24 | v100-2 | 9 |
| C28 | v100-1 | 4 |

GPU 4 has about 1.1 GiB existing allocation and 31.4 GiB free. Do not terminate
its existing owner or clear GPU state. Its launcher tolerates at most 2048 MiB
existing allocation and still requires at least 30000 MiB free at every launch.
Other assignments require at most 256 MiB existing allocation. Recheck before
each child process. Report/log prefix: `runs/s1d_C_dimensions_v1`.

At launch GPU 1 acquired other processes, so C20's preflight refused to start.
That failed launcher report is preserved. C20 is queued under prefix
`runs/s1d_C_dimensions_v1_C20retry`, waiting until GPU 1 meets the original
idle/free-memory limits. C24/C28 use the original prefix.
