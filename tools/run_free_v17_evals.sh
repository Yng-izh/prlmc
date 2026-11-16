#!/usr/bin/env bash
# No-training V17 evaluations on the V13 checkpoint, one mechanism per run.
#
#   density   1,620 core slots re-split per block by redundancy      (V13Density - V13)
#   absorb    fixed 180/block, near-duplicate evictees absorbed      (V13Density - V13)
#   attn_q90  graft keys pulled down to the core p90 key norm        (V17 - V13)
#   attn_q50  graft keys pulled down to the core median key norm     (V17 - V13)
#
# Two GPU queues run in parallel; each job does a 2-row smoke run first and only starts
# the 158-row run when the smoke run finished. Usage (from the repo root):
#   nohup bash tools/run_free_v17_evals.sh > runs/free_v17/runner.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/data/zhangjiayang/yyz/env/bin/python
TAG=${TAG:-free_v17}
RUN=runs/$TAG
OUT=reports/cross_block
mkdir -p "$RUN" "$OUT"
STATUS=$RUN/status.txt
ABSORB_COS=${ABSORB_COS:-0.75}

COMMON=(--data-dir /data/zhangjiayang/yyz/datasets/LongBench/extracted/data
        --task hotpotqa
        --supporting-facts-glob '/data/zhangjiayang/yyz/datasets/hotpot_qa/distractor/validation-*.parquet'
        --model-path /dev/shm/s1d_reader_fp16
        --checkpoint runs/natural_merge_v9/full_v10.pt
        --anchor runs/anchor_residual/anchor_only_rank64.pt
        --v11 runs/cross_block/plain.pt
        --v11-evict runs/cross_block/cross_lr.pt)

job_args() {
    case "$1" in
        density)  echo "--v17-window-density" ;;
        absorb)   echo "--v17-window-density --v17-window-quota fixed --v17-merge-cos $ABSORB_COS" ;;
        attn_q90) echo "--v17 runs/cross_block/v13_attn_q90.pt --v17-no-attn" ;;
        attn_q50) echo "--v17 runs/cross_block/v13_attn_q50.pt --v17-no-attn" ;;
        *) echo "unknown job $1" >&2; return 1 ;;
    esac
}

note() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

# The eval script's main() returns the report dict and `raise SystemExit(dict)` exits 1
# even on success, so success is read from the report itself, never from the exit code.
complete() {
    [ -s "$1" ] && "$PY" -c "import json,sys; sys.exit(json.load(open(sys.argv[1])).get('status') != 'complete')" "$1"
}

# runtime.load_runtime hard-codes cuda:0, so the GPU is chosen by masking and the script
# then sees exactly one device.
evaluate() {
    local name=$1 gpu=$2 out=$3 log=$4; shift 4
    local extra
    extra=$(job_args "$name") || return 1
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu "$PY" tools/eval_cross_block_calibration.py \
        "${COMMON[@]}" $extra --cuda-device 0 "$@" --out "$out" > "$log" 2>&1
    complete "$out"
}

smoke() {
    local name=$1 gpu=$2
    if [ -e "$OUT/${TAG}_$name.json" ]; then
        note "$name: $OUT/${TAG}_$name.json already exists; skipped (move it to rerun)"
        return 1
    fi
    rm -f "$RUN/smoke_$name.json"     # this runner's own scratch file
    note "$name gpu$gpu smoke start"
    if ! evaluate "$name" "$gpu" "$RUN/smoke_$name.json" "$RUN/smoke_$name.log" \
            --max-samples 2; then
        note "$name gpu$gpu smoke FAILED (see $RUN/smoke_$name.log)"
        return 1
    fi
    # V13 must match the reference rows, and V17NoAttn must equal V13.
    if ! "$PY" tools/postcheck_free_evals.py --out "$RUN/smoke_$name.check.json" \
            "$RUN/smoke_$name.json" > /dev/null 2>&1 \
       || ! "$PY" -c "import json,sys; sys.exit(not json.load(open(sys.argv[1]))[0].get('trustworthy'))" \
            "$RUN/smoke_$name.check.json"; then
        note "$name gpu$gpu smoke ran but FAILED its checks (see $RUN/smoke_$name.check.json)"
        return 1
    fi
    note "$name gpu$gpu smoke ok"
}

full() {
    local name=$1 gpu=$2
    note "$name gpu$gpu full start"
    if evaluate "$name" "$gpu" "$OUT/${TAG}_$name.json" "$RUN/$name.log" \
            --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json; then
        note "$name gpu$gpu full done -> $OUT/${TAG}_$name.json"
    else
        note "$name gpu$gpu full FAILED (see $RUN/$name.log)"
        return 1
    fi
}

# Smoke every job of the queue first, so a broken arm is found in minutes, not hours.
queue() {
    local gpu=$1; shift
    local passed=() name
    for name in "$@"; do smoke "$name" "$gpu" && passed+=("$name"); done
    for name in "${passed[@]}"; do full "$name" "$gpu"; done
}

note "runner start pid $$ absorb_cos=$ABSORB_COS"
if [ ! -d /dev/shm/s1d_reader_fp16 ]; then
    note "reader /dev/shm/s1d_reader_fp16 missing; abort"; exit 1
fi
for pair in q90:0.9 q50:0.5; do
    tag=${pair%%:*}; q=${pair#*:}
    "$PY" tools/wrap_v13_as_v17.py --v13 runs/cross_block/cross_lr.pt \
        --norm-quantile "$q" --out "runs/cross_block/v13_attn_$tag.pt" >> "$STATUS" 2>&1 \
        || { note "wrap $q failed; abort"; exit 1; }
done

# Override to rerun a subset, e.g. QUEUE0=attn_q90 QUEUE1=attn_q50.
# shellcheck disable=SC2086
queue 0 ${QUEUE0:-density attn_q90} &
# shellcheck disable=SC2086
queue 1 ${QUEUE1:-absorb attn_q50} &
wait

"$PY" tools/postcheck_free_evals.py --out "$RUN/postcheck.json" \
    "$OUT/${TAG}_density.json" "$OUT/${TAG}_absorb.json" \
    "$OUT/${TAG}_attn_q90.json" "$OUT/${TAG}_attn_q50.json" > "$RUN/postcheck.log" 2>&1
note "runner done; summary in $RUN/postcheck.json"
