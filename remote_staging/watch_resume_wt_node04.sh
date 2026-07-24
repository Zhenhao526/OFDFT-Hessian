#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
log=$root/runs/al/free_energy_wt/node04_recovery.log
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
abacus=$root/external/abacus-develop/build-gpu-gcc13-openmp/source/abacus_ml_gpu
config=$root/config/abacus_wt_node04_cpu12_stress.json

mkdir -p "$(dirname "$log")"
printf '%s recovery_watcher_started\n' "$(date -Iseconds)" >> "$log"

until ssh -o BatchMode=yes -o ConnectTimeout=5 node04 true 2>/dev/null; do
    printf '%s waiting_for_node04_ssh\n' "$(date -Iseconds)" >> "$log"
    sleep 60
done
printf '%s node04_ssh_recovered\n' "$(date -Iseconds)" >> "$log"

sessions=(wt_s750_170 wt_s750_174 wt_s750_178 wt_lp1886)
for session in "${sessions[@]}"; do
    if ssh node04 "tmux has-session -t '=$session'" 2>/dev/null; then
        printf '%s existing_session=%s_no_duplicate_resume\n' "$(date -Iseconds)" "$session" >> "$log"
        exit 0
    fi
done

solid_out=$root/runs/al/free_energy_wt/nvt_volume_scan/solid_T0750_recovered_from_vpa17p4
solid_source=$root/runs/al/free_energy_wt/nvt_volume_scan/solid_T0750_from_wt300/vpa_17p400/OUT.al108_solid_T0750_vpa_17p400/MD_dump
if [[ ! -f "$solid_out/manifest.json" ]]; then
    env PYTHONPATH="$root" python3 "$root/scripts/prepare_al108_volume_scan.py" prepare-md \
        --out "$solid_out" \
        --source "$solid_source" --source-frame last \
        --phase solid --temperature 750 \
        --volumes-per-atom 17.0 17.4 17.8 \
        --steps 150 --dt 1 --csvr-tau 10 --dumpfreq 5 --restartfreq 50 \
        --seed 7551 --config "$config" >> "$log" 2>&1
fi

liquid_out=$root/runs/al/free_energy_wt/production/liquid_T0980_vpa18p86_tau5_recovered
liquid_source=$root/runs/al/free_energy_wt/production/liquid_T0980_vpa18p86_tau5/vpa_18p860/OUT.al108_liquid_T0980_vpa_18p860/MD_dump
if [[ ! -f "$liquid_out/manifest.json" ]]; then
    env PYTHONPATH="$root" python3 "$root/scripts/prepare_al108_volume_scan.py" prepare-md \
        --out "$liquid_out" \
        --source "$liquid_source" --source-frame last \
        --phase liquid --temperature 980 --volumes-per-atom 18.86 \
        --steps 250 --dt 1 --csvr-tau 5 --dumpfreq 5 --restartfreq 50 \
        --seed 9851 --config "$config" >> "$log" 2>&1
fi

launch() {
    local session=$1 run_dir=$2 cpus=$3 ranks=$4
    ssh node04 "tmux new-session -d -s '$session' \
        'cd $run_dir && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        taskset -c $cpus $mpirun -np $ranks --bind-to none $abacus > run.stdout 2>&1'"
    printf '%s launched=%s cpus=%s ranks=%s\n' "$(date -Iseconds)" "$session" "$cpus" "$ranks" >> "$log"
}

launch wt_s750r_170 "$solid_out/vpa_17p000" 0-11 12
launch wt_s750r_174 "$solid_out/vpa_17p400" 12-23 12
launch wt_s750r_178 "$solid_out/vpa_17p800" 24-35 12
launch wt_lp1886r "$liquid_out/vpa_18p860" 36-59 24
printf '%s recovery_launch_completed\n' "$(date -Iseconds)" >> "$log"
