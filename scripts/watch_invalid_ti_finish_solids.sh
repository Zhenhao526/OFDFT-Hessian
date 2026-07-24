#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s TI_ROOT LAUNCHER_PID\n' "$0" >&2
    exit 2
fi

ti_root=$1
launcher_pid=$2
log=$ti_root/invalid_liquid_reference_guard.log

step_of() {
    local point=$1
    local md_log
    md_log=$(find "$point" -type f -name running_md.log | head -n 1)
    if [[ -z $md_log ]]; then
        printf '0\n'
        return
    fi
    awk '/STEP OF MOLECULAR DYNAMICS:/{step=$5} END{print step+0}' "$md_log"
}

printf '%s guard_started launcher_pid=%s reason=liquid_reference_rmin_lt_2A\n' \
    "$(date -Iseconds)" "$launcher_pid" >> "$log"

while true; do
    complete=1
    progress=()
    for point in "$ti_root"/solid/lambda_*; do
        step=$(step_of "$point")
        progress+=("$(basename "$point")=$step")
        if (( step < 1000 )); then
            complete=0
        fi
    done
    printf '%s solid_progress %s\n' "$(date -Iseconds)" "${progress[*]}" >> "$log"
    if (( complete )); then
        break
    fi
    sleep 60
done

# Let ABACUS flush its final restart and trajectory files before removing the
# stopped parent shell. A zombie child is already complete and needs no wait.
while true; do
    active=0
    while read -r child; do
        [[ -n $child ]] || continue
        state=$(ps -o stat= -p "$child" 2>/dev/null | xargs || true)
        if [[ -n $state && $state != Z* ]]; then
            active=1
        fi
    done < <(pgrep -P "$launcher_pid" || true)
    (( active == 0 )) && break
    sleep 10
done

if kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM "$launcher_pid" 2>/dev/null || true
    sleep 2
fi
if kill -0 "$launcher_pid" 2>/dev/null; then
    kill -KILL "$launcher_pid" 2>/dev/null || true
fi
tmux kill-session -t wt_phase_ti900 2>/dev/null || true

printf '%s invalidated liquid_lambda0_min_rmin_A=1.821094524 liquid_anchor_gate_A=1.5 required_gate_A=2.0 solid_windows_preserved=true liquid_wave2_started=false\n' \
    "$(date -Iseconds)" > "$ti_root/INVALID_LIQUID_REFERENCE_RMIN_LT_2A"
printf '%s guard_complete\n' "$(date -Iseconds)" >> "$log"
