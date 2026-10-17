#!/bin/bash
# E6b-mini: does one shared carrier-residual checkpoint generalise across hard windows?
# 8 training hard windows, 2 held-out hard windows, 19->9 blocks, structure/loss unchanged.
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
  --out /tmp/e6b_mini_report.json \
  --ckpt /tmp/e6b_mini_ckpt.pt \
  > /tmp/e6b_mini.log 2>&1
echo "EXIT=$?" >> /tmp/e6b_mini.log
