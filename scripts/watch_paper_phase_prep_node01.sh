#!/usr/bin/env bash
set -euo pipefail

root=${1:-/scratch/xzh/OFDFT-Hessian}
base="$root/runs/al/method_paper/phase_prep"
solid=${SOLID_RUN:-"$base/mpn_Al864_solid_T0850_nvt_csvr200_pristine_node01"}
liquid=${LIQUID_RUN:-"$base/mpn_Al864_liquid_T0980_nvt_csvr200_from_T1600_node01"}
log=${WATCH_LOG:-"$base/watch_phase_prep.log"}
expected_steps=${EXPECTED_STEPS:-200}
solid_session=${SOLID_SESSION:-mpn_solid850_csvr}
liquid_session=${LIQUID_SESSION:-mpn_liquid980_csvr}

step_count() {
    local run_dir=$1
    local md_log
    md_log=$(find "$run_dir" -path '*/running_md.log' -print -quit 2>/dev/null || true)
    if [[ -z "$md_log" ]]; then
        printf '0'
        return
    fi
    grep -c 'STEP OF MOLECULAR DYNAMICS' "$md_log" || true
}

session_alive() {
    tmux has-session -t "$1" 2>/dev/null
}

while true; do
    solid_step=$(step_count "$solid")
    liquid_step=$(step_count "$liquid")
    solid_alive=no
    liquid_alive=no
    session_alive "$solid_session" && solid_alive=yes
    session_alive "$liquid_session" && liquid_alive=yes
    printf '%s solid_step=%s solid_alive=%s liquid_step=%s liquid_alive=%s\n' \
        "$(date '+%F %T %Z')" "$solid_step" "$solid_alive" "$liquid_step" "$liquid_alive" >> "$log"

    if [[ "$solid_alive" == no && "$liquid_alive" == no ]]; then
        if (( solid_step >= expected_steps )); then
            cd "$root"
            python3 scripts/analyze_phase_run.py "$solid" --expected solid \
                --out "$solid/phase_analysis.json" >> "$log" 2>&1
        else
            printf '%s ERROR solid stopped before %s steps\n' "$(date '+%F %T %Z')" "$expected_steps" >> "$log"
        fi
        if (( liquid_step >= expected_steps )); then
            cd "$root"
            python3 scripts/analyze_phase_run.py "$liquid" --expected liquid \
                --out "$liquid/phase_analysis.json" >> "$log" 2>&1
        else
            printf '%s ERROR liquid stopped before %s steps\n' "$(date '+%F %T %Z')" "$expected_steps" >> "$log"
        fi
        printf '%s watcher_complete\n' "$(date '+%F %T %Z')" >> "$log"
        exit 0
    fi
    sleep 600
done
