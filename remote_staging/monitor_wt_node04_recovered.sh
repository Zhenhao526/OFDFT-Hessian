#!/usr/bin/env bash
set -u

root=/scratch/xzh/OFDFT-Hessian
solid_root=$root/runs/al/free_energy_wt/nvt_volume_scan/solid_T0750_recovered_from_vpa17p4
liquid_root=$root/runs/al/free_energy_wt/production/liquid_T0980_vpa18p86_tau5_recovered
abacus=$root/external/abacus-develop/build-gpu-gcc13-openmp/source/abacus_ml_gpu

printf 'TIME %s\n' "$(date -Iseconds)"
uptime
findmnt /scratch || true

echo TMUX
tmux ls 2>&1 || true

echo RANKS
mapfile -t pids < <(pgrep -f "$abacus" || true)
printf 'count=%d\n' "${#pids[@]}"
for pid in "${pids[@]}"; do
    [[ -r /proc/$pid/status ]] || continue
    cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    cpus=$(awk '/Cpus_allowed_list:/ {print $2}' "/proc/$pid/status")
    psr=$(ps -o psr= -p "$pid" 2>/dev/null | xargs)
    printf 'pid=%s psr=%s cpus=%s cwd=%s\n' "$pid" "$psr" "$cpus" "$cwd"
done

dirs=(
    "$solid_root/vpa_17p000"
    "$solid_root/vpa_17p400"
    "$solid_root/vpa_17p800"
    "$liquid_root/vpa_18p860"
)

for run_dir in "${dirs[@]}"; do
    echo "RUN $run_dir"
    if [[ ! -d $run_dir ]]; then
        echo MISSING
        continue
    fi
    find "$run_dir" -maxdepth 3 -type f -printf '%T@|%s|%p\n' 2>/dev/null \
        | sort -n | tail -n 30
    if [[ -f $run_dir/run.stdout ]]; then
        echo STDOUT_TAIL
        tail -n 20 "$run_dir/run.stdout"
    fi
    logfile=$(find "$run_dir" -path '*/OUT.*/running_md.log' -type f -print -quit 2>/dev/null || true)
    if [[ -n $logfile ]]; then
        echo "MDLOG $logfile"
        tail -n 35 "$logfile"
    fi
done
