#!/usr/bin/env bash
# Run the remaining 2x2 cells strictly sequentially on one GPU.
# Cell A is expected to be running already; wait for it, then do B and C.
cd /share/latent/PRLM-C
echo "=== chaining b, c after a ==="
while pgrep -f "batch_run.py" > /dev/null; do sleep 60; done
echo "=== a done at $(date) ==="
for CELL in b c; do
  echo "=== launching cell $CELL at $(date) ==="
  CELL=$CELL GPU=6 QUERIES=0 bash run_2x2.sh
done
echo "=== 2x2 matrix complete at $(date) ==="
