#!/usr/bin/env bash
set -euo pipefail

WS=/home/shenwei01/WT_Mg_melting_workspace_20260804
ROOT=runs/mg_three_method_20260804/xwm/ksdft_multitemperature_oneway_diagnostic_v1

for node in node02 node06; do
  phase=solid
  [[ "$node" == node06 ]] && phase=liquid
  echo "== $node $phase =="
  ssh "$node" "cd $WS/$ROOT; \
    printf 'done='; find jobs -name sp.done | wc -l; \
    printf 'failed='; find jobs -name sp.failed | wc -l; \
    printf 'workers='; pgrep -x abacus_pw_para | wc -l; \
    printf 'prterun='; pgrep -x prterun | wc -l; \
    printf 'final_energies='; grep -R -l '!FINAL_ETOT_IS' jobs/T*/$phase/step0775/OUT.*/running_scf.log 2>/dev/null | wc -l; \
    printf 'entropy_rows='; grep -R -l 'E_entropy(-TS)' jobs/T*/$phase/step0775/OUT.*/running_scf.log 2>/dev/null | wc -l; \
    printf 'fatal_markers='; grep -R -E -l 'SCF IS NOT CONVERGED|NaN|nan|FATAL|ERROR' jobs/T*/$phase/step0775/OUT.*/running_scf.log 2>/dev/null | wc -l; \
    for log in jobs/T*/$phase/step0775/OUT.*/running_scf.log; do \
      [[ -f \"\$log\" ]] || continue; \
      printf '%s ' \"\$log\"; \
      grep '#ELEC ITER#' \"\$log\" | tail -1 || true; \
      grep -E 'E_entropy\\(-TS\\)|!FINAL_ETOT_IS' \"\$log\" | tail -2 || true; \
    done; \
    echo 'active_production_logs:'; \
    find jobs -path '*/OUT.*/running_scf.log' -mmin -15 -print | sort | while read -r log; do \
      job=\$(dirname \"\$(dirname \"\$log\")\"); \
      [[ -f \"\$job/sp.done\" ]] && continue; \
      printf '%s ' \"\$log\"; grep '#ELEC ITER#' \"\$log\" | tail -1 || true; \
    done; \
    printf 'masks='; for pid in \$(pgrep -x abacus_pw_para | head -2); do taskset -pc \"\$pid\" | sed 's/.*: //'; done | sort -u | tr '\n' ','; echo; \
    tail -8 pilot_${node}.runner.log 2>/dev/null || true"
done
