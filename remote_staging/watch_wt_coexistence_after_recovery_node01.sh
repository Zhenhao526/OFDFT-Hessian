#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
pipeline_root=${PIPELINE_ROOT:-$repo/runs/al/free_energy_wt/melting_pipeline_pairv2_recovery_T1100_v2_node01}
analysis_root=${ANALYSIS_ROOT:-$repo/runs/al/free_energy_wt/fusion_enthalpy_sampling/T0900_T0975_T1050_adaptive_T1100_pairv2_steps3000_v2_node01}
validation_root=${VALIDATION_ROOT:-$repo/runs/al/free_energy_wt/direct_coexistence_validation/pairv2_recovery_T1100_v2_node01}

pipeline_done=$pipeline_root/pipeline.done
recovery_failed=$analysis_root/enthalpy_recovery.failed
melting=$analysis_root/gibbs_helmholtz_melting.json
handoff_log=$validation_root/recovery_handoff.log

mkdir -p "$validation_root"
printf '%s waiting_for_final_verified_melting\n' "$(date -Iseconds)" > "$handoff_log"
while [[ ! -e $pipeline_done ]]; do
    if [[ -e $recovery_failed ]]; then
        printf '%s refusing_validation reason=enthalpy_recovery_failed\n' \
            "$(date -Iseconds)" | tee -a "$handoff_log"
        exit 2
    fi
    sleep 60
done

if [[ ! -e $melting ]]; then
    printf '%s refusing_validation reason=missing_melting_result\n' \
        "$(date -Iseconds)" | tee -a "$handoff_log"
    exit 2
fi
printf '%s launching_independent_validation\n' "$(date -Iseconds)" \
    | tee -a "$handoff_log"
exec env PIPELINE_DONE="$pipeline_done" MELTING="$melting" \
    VALIDATION_ROOT="$validation_root" \
    bash "$repo/scripts/watch_wt_coexistence_validation_node01.sh"
