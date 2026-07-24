#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/scratch/xzh/OFDFT-Hessian}
base=${BASE:-$root/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_lambda5_steps1000_v1_node01}
launcher_log=$base/run_phase_specific_ti_node01.log
analysis_log=$base/post_ti_analysis.log
done_file=$base/post_ti_analysis.done

if [[ -e $done_file ]]; then
    printf '%s analysis_already_complete done_file=%s\n' "$(date -Iseconds)" "$done_file"
    exit 0
fi

printf '%s waiting_for_ti launcher_log=%s\n' "$(date -Iseconds)" "$launcher_log" \
    > "$analysis_log"
while ! grep -q 'phase_specific_ti_complete' "$launcher_log" 2>/dev/null; do
    sleep 60
done

completion=$(grep 'phase_specific_ti_complete' "$launcher_log" | tail -1)
printf '%s observed_completion record=%s\n' "$(date -Iseconds)" "$completion" \
    | tee -a "$analysis_log"
if [[ $completion != *'failed=0'* ]]; then
    printf '%s refusing_analysis reason=launcher_failed\n' "$(date -Iseconds)" \
        | tee -a "$analysis_log"
    exit 2
fi

cd "$root"
for phase in solid liquid; do
    for percent in 25 50 75; do
        fraction=$(printf '0.%02d' "$percent")
        output=$base/$phase/ti_prod_d${percent}.json
        printf '%s analysis_started phase=%s discard=%s output=%s\n' \
            "$(date -Iseconds)" "$phase" "$fraction" "$output" | tee -a "$analysis_log"
        env PYTHONPATH="$root" python3 scripts/analyze_wt_pair_ti_production.py \
            "$base/$phase" --discard-fraction "$fraction" --out "$output" \
            >> "$analysis_log" 2>&1
        printf '%s analysis_finished phase=%s discard=%s\n' \
            "$(date -Iseconds)" "$phase" "$fraction" | tee -a "$analysis_log"
    done
    summary=$base/$phase/discard_convergence_summary.json
    printf '%s convergence_summary_started phase=%s output=%s\n' \
        "$(date -Iseconds)" "$phase" "$summary" | tee -a "$analysis_log"
    env PYTHONPATH="$root" python3 scripts/summarize_wt_ti_convergence.py \
        "$base/$phase/ti_prod_d25.json" \
        "$base/$phase/ti_prod_d50.json" \
        "$base/$phase/ti_prod_d75.json" \
        --out "$summary" >> "$analysis_log" 2>&1
    printf '%s convergence_summary_finished phase=%s\n' \
        "$(date -Iseconds)" "$phase" | tee -a "$analysis_log"
done

printf '%s post_ti_analysis_complete\n' "$(date -Iseconds)" | tee -a "$analysis_log"
date -Iseconds > "$done_file"
