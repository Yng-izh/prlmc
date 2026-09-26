#!/usr/bin/env bash
# Wait for the PRLM-C observation run to finish, then print the calibration.
#
# Polls instead of holding one SSH session open: the run is detached with
# setsid, so it survives a dropped connection, and a 3.5-hour single connection
# would almost certainly drop. Each iteration is a fresh short call.
#
# Reports the raw per-write signal distributions rather than a pass/fail, because
# the point of the observation is to choose thresholds, and that is a judgement
# made against the numbers.

HOST="root@150.158.54.163"
KEY="$HOME/.ssh/keys/yyz.pem"
PORT=7001
STATS=/share/latent/PRLM-C/outputs/prlmc-obs/stats/prlmc_stats.jsonl
PY=/share/latent/mllm_env/bin/python

ssh_do() {
  ssh -i "$KEY" -p "$PORT" -o ConnectTimeout=30 -o StrictHostKeyChecking=no \
      -o ServerAliveInterval=20 "$HOST" "$@"
}

started=$(date +%s)
while true; do
  alive=$(ssh_do 'pgrep -fc batch_run.py || true' 2>/dev/null | tr -d '[:space:]')
  lines=$(ssh_do "wc -l < $STATS 2>/dev/null || echo 0" 2>/dev/null | tr -d '[:space:]')
  elapsed=$(( ($(date +%s) - started) / 60 ))
  echo "[monitor] ${elapsed}m elapsed  batch_procs=${alive:-?}  stats_lines=${lines:-0}"

  if [ "${alive:-0}" = "0" ]; then
    echo "[monitor] run finished; collecting calibration"
    break
  fi
  sleep 180
done

echo
echo "===================== calibration ====================="
ssh_do "cat $STATS 2>/dev/null | wc -l"
ssh_do "$PY /share/latent/PRLM-C/tools/calibrate_obs.py $STATS" 2>&1

echo
echo "===================== run tail ========================"
ssh_do 'tail -25 /share/latent/PRLM-C/logs/obs_run2.log'
