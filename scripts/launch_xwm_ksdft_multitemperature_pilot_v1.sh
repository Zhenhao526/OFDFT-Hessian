#!/usr/bin/env bash
set -euo pipefail

WS=/home/shenwei01/WT_Mg_melting_workspace_20260804
ROOT=runs/mg_three_method_20260804/xwm/ksdft_multitemperature_oneway_diagnostic_v1
RUNNER=repository/scripts/run_xwm_ksdft_multitemperature_lane.sh

for node in node02 node06; do
  echo "== $node preflight =="
  ssh "$node" 'if pgrep -x abacus_pw_para >/dev/null || pgrep -x prterun >/dev/null; then echo BUSY; exit 1; fi; df -h /home | tail -1; tmux list-sessions 2>/dev/null || true'
done

ssh node02 "tmux new-session -d -s mg_xwm_ksmt_pilot_v1_node02 \"cd $WS && $RUNNER $ROOT $ROOT/manifests/pilot_node02.txt > $ROOT/pilot_node02.runner.log 2>&1\""
ssh node06 "tmux new-session -d -s mg_xwm_ksmt_pilot_v1_node06 \"cd $WS && $RUNNER $ROOT $ROOT/manifests/pilot_node06.txt > $ROOT/pilot_node06.runner.log 2>&1\""

sleep 12
for node in node02 node06; do
  echo "== $node running =="
  ssh "$node" 'tmux list-sessions 2>/dev/null || true; printf "workers="; pgrep -x abacus_pw_para | wc -l; printf "prterun="; pgrep -x prterun | wc -l; for pid in $(pgrep -x abacus_pw_para | head -2); do taskset -pc "$pid"; done'
done
