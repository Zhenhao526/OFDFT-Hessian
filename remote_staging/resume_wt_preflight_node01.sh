#!/usr/bin/env bash
set -euo pipefail

state=/proc/fs/beegfs/c22801-6A1925D1-node01/storage_target_state
repo=/scratch/xzh/OFDFT-Hessian
analysis=$repo/runs/al/free_energy_wt/fusion_enthalpy_sampling/T0900_T0975_T1050_adaptive_T1100_pairv2_steps3000_v2_node01
t1100=$analysis/critical_extensions/T1100_steps3000_round2
binary=$repo/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun

if grep -q 'Offline' "$state"; then
    echo "NOT READY: BeeGFS has offline targets."
    grep 'Offline' "$state"
    exit 10
fi

if pgrep -af 'wt_enthalpy|wt_coexist|T1100_steps3000_round2|abacus_wt_ti_cpu' \
    | grep -v resume_wt_preflight_node01.sh; then
    echo "NOT READY: project processes are already present."
    exit 11
fi

required=(
    "$repo/runs/al/free_energy_wt/melting_free_energy_T0900_pairv2_v2.json"
    "$analysis/critical_extensions/T0975_steps3000_round2/confirmation_summary.json"
    "$analysis/critical_extensions/T1050_steps3000_round2/confirmation_summary.json"
    "$binary"
    "$mpirun"
)
for path in "${required[@]}"; do
    if [[ ! -r $path ]]; then
        echo "NOT READY: unreadable required path: $path"
        exit 12
    fi
done

for phase in solid liquid; do
    source=$(find "$t1100/$phase" -type f -name 'STRU_MD_1350' -print -quit)
    if [[ -z $source || ! -r $source ]]; then
        echo "NOT READY: missing readable $phase STRU_MD_1350"
        exit 13
    fi
    velocity_count=$(grep -c ' v ' "$source")
    if [[ $velocity_count -ne 108 ]]; then
        echo "NOT READY: $phase STRU_MD_1350 has $velocity_count velocities, expected 108"
        exit 14
    fi
    echo "READY SOURCE: $phase $source velocities=$velocity_count"
done

echo "PREFLIGHT PASSED"
echo "Next: prepare a new 1650-step T1100 continuation from both STRU_MD_1350 files."
echo "Do not launch the old round2 directory or randomize velocities."
