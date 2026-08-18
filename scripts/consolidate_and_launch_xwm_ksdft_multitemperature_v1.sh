#!/usr/bin/env bash
set -euo pipefail

WS=/home/shenwei01/WT_Mg_melting_workspace_20260804
ROOT=runs/mg_three_method_20260804/xwm/ksdft_multitemperature_oneway_diagnostic_v1
VALIDATOR=repository/scripts/validate_xwm_ksdft_multitemperature_pilot.py
RUNNER=repository/scripts/run_xwm_ksdft_multitemperature_lane.sh

cd "$WS"

for temperature in 0950 1000 1050; do
  rsync -a "node02:$WS/$ROOT/jobs/T${temperature}/solid/step0775/" "$ROOT/jobs/T${temperature}/solid/step0775/"
  rsync -a "node06:$WS/$ROOT/jobs/T${temperature}/liquid/step0775/" "$ROOT/jobs/T${temperature}/liquid/step0775/"
done
rsync -a "node02:$WS/$ROOT/PILOT_VALIDATION_SOLID_NODE02.json" "$ROOT/"
rsync -a "node06:$WS/$ROOT/PILOT_VALIDATION_LIQUID_NODE06.json" "$ROOT/"

PYTHONPATH=repository python3 "$VALIDATOR" "$ROOT" --output "$ROOT/PILOT_VALIDATION_COMBINED.json"
sha256sum "$ROOT/PILOT_VALIDATION_COMBINED.json" > "$ROOT/PILOT_VALIDATION_COMBINED.sha256"

for node in node02 node06; do
  for temperature in 0950 1000 1050; do
    rsync -a "$ROOT/jobs/T${temperature}/solid/step0775/" "$node:$WS/$ROOT/jobs/T${temperature}/solid/step0775/"
    rsync -a "$ROOT/jobs/T${temperature}/liquid/step0775/" "$node:$WS/$ROOT/jobs/T${temperature}/liquid/step0775/"
  done
  rsync -a "$ROOT/PILOT_VALIDATION_COMBINED.json" "$ROOT/PILOT_VALIDATION_COMBINED.sha256" "$node:$WS/$ROOT/"
  ssh "$node" "cd $WS && PYTHONPATH=repository python3 $VALIDATOR $ROOT --output $ROOT/PILOT_VALIDATION_COMBINED_${node}.json"
done

for node in node02 node06; do
  echo "== $node production preflight =="
  ssh "$node" 'if pgrep -x abacus_pw_para >/dev/null || pgrep -x prterun >/dev/null; then echo BUSY; exit 1; fi; df -h /home | tail -1; tmux list-sessions 2>/dev/null || true'
done

ssh node02 "tmux new-session -d -s mg_xwm_ksmt_prod_v1_node02 \"cd $WS && $RUNNER $ROOT $ROOT/manifests/production_node02.txt > $ROOT/production_node02.runner.log 2>&1\""
ssh node06 "tmux new-session -d -s mg_xwm_ksmt_prod_v1_node06 \"cd $WS && $RUNNER $ROOT $ROOT/manifests/production_node06.txt > $ROOT/production_node06.runner.log 2>&1\""

sleep 12
for node in node02 node06; do
  echo "== $node production running =="
  ssh "$node" 'tmux list-sessions 2>/dev/null || true; printf "workers="; pgrep -x abacus_pw_para | wc -l; printf "prterun="; pgrep -x prterun | wc -l; for pid in $(pgrep -x abacus_pw_para | head -2); do taskset -pc "$pid"; done'
done
