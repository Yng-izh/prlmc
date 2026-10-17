#!/bin/bash
PY=/share/latent/mllm_env/bin/python
cd /share/latent/PRLM-C
run() {
  CUDA_VISIBLE_DEVICES=7 $PY tools/longbench_prlmc_eval.py --data-dir /share/latent/benchmarks/LongBench/data --out runs/longbench_prlmc_v4/$2 --total-tokens 8192 --cuda-device 7 --arm Selection --tasks $3 --task-range $4 $5 --resume > runs/longbench_prlmc_v4/logs_$2.log 2>&1
}
run 7 sel8k__qasper_100_200.json qasper 100 200
run 7 sel8k__hotpotqa_100_200.json hotpotqa 100 200
run 7 sel8k__mfq4__multifieldqa_en_114_150.json multifieldqa_en 114 150
run 7 sel8k__narrativeqa_100_200.json narrativeqa 100 200
