#!/usr/bin/env bash
set -u

root=/scratch/xzh/OFDFT-Hessian
base=$root/runs/al/free_energy/mpn_samples
solid=$base/mpn_Al108_solid_T0850_nvt_csvr500_v2
liquid=$base/mpn_Al108_liquid_T1600_nvt_csvr1000
log=$base/watch_al108_free_energy_node04.log
exec >>"$log" 2>&1

frame_count() {
    local run_dir=$1
    local dump
    dump=$(find "$run_dir" -maxdepth 2 -name MD_dump -print -quit)
    if [[ -z "$dump" ]]; then
        printf '0'
    else
        grep -c '^MDSTEP:' "$dump"
    fi
}

while tmux has-session -t fe_solid108_850 2>/dev/null || \
      tmux has-session -t fe_liquid108_1600 2>/dev/null; do
    printf '%s solid_frames=%s liquid_frames=%s\n' \
        "$(date --iso-8601=seconds)" \
        "$(frame_count "$solid")" \
        "$(frame_count "$liquid")"
    sleep 300
done

cd "$root" || exit 1
PYTHONPATH=. /usr/bin/python3 scripts/analyze_phase_run.py \
    --expected solid \
    --out "$solid/phase_analysis.json" \
    "$solid"
solid_status=$?

PYTHONPATH=. /usr/bin/python3 scripts/analyze_phase_run.py \
    --expected liquid \
    --out "$liquid/phase_analysis.json" \
    "$liquid"
liquid_status=$?

printf '%s solid_analysis_exit=%s liquid_analysis_exit=%s\n' \
    "$(date --iso-8601=seconds)" "$solid_status" "$liquid_status"
exit $((solid_status != 0 || liquid_status != 0))
