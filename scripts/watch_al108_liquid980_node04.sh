#!/usr/bin/env bash
set -u

root=/scratch/xzh/OFDFT-Hessian
run=$root/runs/al/free_energy/mpn_samples/mpn_Al108_liquid_T0980_nvt_csvr1000_from_T1600_node04
log=$run/watch_liquid980_node04.log
session=fe_liquid108_980

exec >>"$log" 2>&1

frame_count() {
    local dump
    dump=$(find "$run" -maxdepth 2 -name MD_dump -print -quit)
    if [[ -z "$dump" ]]; then
        printf '0'
    else
        grep -c '^MDSTEP:' "$dump"
    fi
}

while tmux has-session -t "=$session" 2>/dev/null; do
    printf '%s frames=%s ranks=%s\n' \
        "$(date --iso-8601=seconds)" \
        "$(frame_count)" \
        "$(pgrep -x abacus_ml_gpu | wc -l)"
    sleep 300
done

printf '%s production_session_finished frames=%s\n' \
    "$(date --iso-8601=seconds)" "$(frame_count)"

cd "$root" || exit 1
PYTHONPATH=. /usr/bin/python3 scripts/analyze_phase_run.py \
    --expected liquid \
    --out "$run/phase_analysis.json" \
    "$run"
status=$?

printf '%s liquid_analysis_exit=%s\n' \
    "$(date --iso-8601=seconds)" "$status"
exit "$status"
