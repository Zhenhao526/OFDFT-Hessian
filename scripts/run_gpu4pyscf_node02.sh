#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "node02" ]]; then
  echo "ERROR: this GPU4PySCF runtime is bound to node02" >&2
  exit 1
fi

NODE_ROOT="${QM9_NODE02_ROOT:-/home/shenwei01/xzh_node02_20260724}"
CODE_ROOT="${QM9_CODE_ROOT:-${NODE_ROOT}/work/structures25}"
GPU4PYSCF_ENV="${GPU4PYSCF_ENV:-${NODE_ROOT}/envs/gpu4pyscf-cuda12-py311}"
GPU4PYSCF_PYTHON="${GPU4PYSCF_PYTHON:-${GPU4PYSCF_ENV}/bin/python}"
MAIN_SITE="${CODE_ROOT}/.venv/lib/python3.11/site-packages"

if [[ ! -x "${GPU4PYSCF_PYTHON}" ]]; then
  echo "ERROR: GPU4PySCF Python is missing: ${GPU4PYSCF_PYTHON}" >&2
  exit 1
fi

cuda_library_dirs=()
for library_dir in "${MAIN_SITE}"/nvidia/*/lib; do
  if [[ -d "${library_dir}" ]]; then
    cuda_library_dirs+=("${library_dir}")
  fi
done
if [[ "${#cuda_library_dirs[@]}" -eq 0 ]]; then
  echo "ERROR: CUDA 12 runtime libraries are missing below ${MAIN_SITE}/nvidia" >&2
  exit 1
fi

cuda_library_path="$(IFS=:; echo "${cuda_library_dirs[*]}")"
export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OPENSSL_CONF="${OPENSSL_CONF:-/etc/ssl/openssl.cnf}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${NODE_ROOT}/cache/cupy}"
export PYSCF_TMPDIR="${PYSCF_TMPDIR:-${NODE_ROOT}/tmp/pyscf}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${CUPY_CACHE_DIR}" "${PYSCF_TMPDIR}"
exec "${GPU4PYSCF_PYTHON}" "$@"
