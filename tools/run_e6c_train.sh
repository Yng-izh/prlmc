#!/bin/bash
# E6c, step 1 of 2: train the shared carrier-residual checkpoint that the
# `prlmc_e6c-carrier` benchmark arm loads.
#
# Identical to the E6b-mini run in every respect except the Stage B budget:
# 96 steps per window instead of 24, so Stage B totals 768 steps instead of 192.
# Nothing else moves -- same 8 hard windows, same Stage A of 4, same 19 -> 9 block
# geometry, same lr 1e-3, same gamma_max 0.05, same loss, same selector. The run is
# therefore a pure step-budget extension and any change in the checkpoint is
# attributable to that alone.
#
# Why 96: the 8 x 24 run ended with `routing_entropy 2.9417140`, which is 0.99907x
# of `ln(19) = 2.9444`, i.e. routing had barely moved off uniform. The single-window
# 220-step run reached 0.93735x with recovery 1/3 -> 2/3. 96 is a large step in that
# direction without paying the 9x cost of 220 per window.
#
# Checkpoint path and report name are the E6c ones. `run_e6c_bench.sh` reads
# /tmp/e6c_ckpt.pt, so do not point --ckpt at the e6b artifact.
cd /tmp || exit 1

E6A_NEED_MB=21000 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PRLMC_MEM_TRACE=0 \
/share/latent/mllm_env/bin/python /tmp/e6b_mini.py \
  --steps-a-per-window 4 \
  --steps-b-per-window 96 \
  --train-windows 8 \
  --eval-windows 2 \
  --probes 20 \
  --eval-probes 60 \
  --max-probe-tokens 256 \
  --lr 1e-3 \
  --gamma-max 0.05 \
  --min-probes 4 \
  --min-lost 2 \
  --max-screen 66 \
  --screen-budget-min 100 \
  --candidates /tmp/e6a_screen_data.json \
  --out /tmp/e6c_train_report.json \
  --ckpt /tmp/e6c_ckpt.pt \
  > /tmp/e6c_train.log 2>&1
echo "EXIT=$?" >> /tmp/e6c_train.log
