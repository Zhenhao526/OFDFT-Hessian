#!/usr/bin/env bash
set -euo pipefail

runtime=/home/shenwei01/wt_melting_runtime_20260724
build=$runtime/build-abacus-wt-cpu
status=$runtime/build_status.txt
binary=$build/source/abacus_pw_para
compat_binary=$build/source/abacus_wt_ti_cpu

trap 'rc=$?; printf "failed rc=%s time=%s\n" "$rc" "$(date -Iseconds)" > "$status"; exit "$rc"' ERR

source "$runtime/activate_wt_local.sh"
export OMPI_CC=/usr/bin/gcc
export OMPI_CXX=/usr/bin/g++

printf 'running time=%s\n' "$(date -Iseconds)" > "$status"
start=$(date +%s)
cmake --build "$build" --parallel 32 >> "$runtime/build.log" 2>&1
end=$(date +%s)

test -x "$binary"
ln -sfn abacus_pw_para "$compat_binary"

{
    printf 'complete time=%s\n' "$(date -Iseconds)"
    printf 'build_seconds=%s\n' "$((end - start))"
    printf 'binary=%s\n' "$binary"
    printf 'compat_binary=%s\n' "$compat_binary"
} > "$status"
