#!/usr/bin/env bash
set -euo pipefail

runtime=/home/shenwei01/wt_melting_runtime_20260724
restore=/home/shenwei01/wt_melting_restore_20260724
source_run=$restore/integrated/prepared_fallback/T1100_resume_from_s1300_to_s3000
run_root=/home/shenwei01/wt_melting_local_runs_20260724/T1100_resume_s1300_to_s3000
build_status=$runtime/build_status.txt
binary=$runtime/build-abacus-wt-cpu/source/abacus_wt_ti_cpu
mpirun=$runtime/conda_prefix/bin/mpirun
pseudo_dir=$restore/integrated/abacus_source/tests/PP_ORB
pipeline_log=$runtime/resume_s1300_pipeline.log
pipeline_status=$runtime/resume_s1300_pipeline_status.txt

trap 'rc=$?; printf "failed rc=%s time=%s\n" "$rc" "$(date -Iseconds)" > "$pipeline_status"; exit "$rc"' ERR

printf 'waiting_for_build time=%s\n' "$(date -Iseconds)" > "$pipeline_status"
for _ in $(seq 1 180); do
    if grep -q '^complete ' "$build_status" 2>/dev/null; then
        break
    fi
    if grep -q '^failed ' "$build_status" 2>/dev/null; then
        exit 20
    fi
    sleep 10
done
grep -q '^complete ' "$build_status"

source "$runtime/activate_wt_local.sh"
export OMPI_CC=/usr/bin/gcc
export OMPI_CXX=/usr/bin/g++
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

test -x "$binary"
test -x "$mpirun"
test -f "$pseudo_dir/al.gga.psp"
if ldd "$binary" | grep -q 'not found'; then
    ldd "$binary" | grep 'not found' >&2
    exit 21
fi

if [[ -e $run_root ]]; then
    printf 'refusing_existing_run_root=%s\n' "$run_root" >&2
    exit 22
fi
mkdir -p "$run_root"
cp -a "$source_run/solid" "$run_root/solid"
cp -a "$source_run/liquid" "$run_root/liquid"
cp "$source_run/resume_manifest.json" "$run_root/resume_manifest.json"

for phase in solid liquid; do
    point=$run_root/$phase
    sed -i "s|^pseudo_dir .*|pseudo_dir $pseudo_dir|" "$point/INPUT"
    grep -q '^md_nstep 1700$' "$point/INPUT"
    grep -q '^init_vel 1$' "$point/INPUT"
    velocity_count=$(grep -c ' v ' "$point/STRU")
    if [[ $velocity_count -ne 108 ]]; then
        printf 'bad_velocity_count phase=%s count=%s\n' "$phase" "$velocity_count" >&2
        exit 23
    fi
done

smoke=$run_root/smoke_solid_one_step
cp -a "$run_root/solid" "$smoke"
sed -i 's/^suffix .*/suffix al108_solid_T1100_resume_smoke/' "$smoke/INPUT"
sed -i 's/^md_nstep 1700$/md_nstep 1/' "$smoke/INPUT"

printf 'smoke_running time=%s\n' "$(date -Iseconds)" > "$pipeline_status"
(
    cd "$smoke"
    taskset -c 0-11 "$mpirun" -np 12 --bind-to none \
        --mca pml ob1 --mca btl self,vader,tcp \
        "$binary" > run.stdout 2>&1
)
grep -q 'STEP OF MOLECULAR DYNAMICS' "$smoke/run.stdout"

printf 'production_running time=%s run_root=%s\n' \
    "$(date -Iseconds)" "$run_root" > "$pipeline_status"
: > "$pipeline_log"
pids=()
points=("$run_root/solid" "$run_root/liquid")
cpu_ranges=(0-35 38-73)
for slot in 0 1; do
    point=${points[$slot]}
    cpus=${cpu_ranges[$slot]}
    (
        cd "$point"
        exec taskset -c "$cpus" "$mpirun" -np 36 --bind-to none \
            --mca pml ob1 --mca btl self,vader,tcp \
            "$binary" > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s cpus=%s pid=%s\n' \
        "$(date -Iseconds)" "$point" "$cpus" "$!" >> "$pipeline_log"
done

failed=0
for slot in 0 1; do
    if wait "${pids[$slot]}"; then
        printf '%s finished point=%s\n' \
            "$(date -Iseconds)" "${points[$slot]}" >> "$pipeline_log"
    else
        rc=$?
        printf '%s failed point=%s rc=%s\n' \
            "$(date -Iseconds)" "${points[$slot]}" "$rc" >> "$pipeline_log"
        failed=1
    fi
done
if [[ $failed -ne 0 ]]; then
    exit 24
fi

printf 'complete time=%s run_root=%s\n' \
    "$(date -Iseconds)" "$run_root" > "$pipeline_status"
