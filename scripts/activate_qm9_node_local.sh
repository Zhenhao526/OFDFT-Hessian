#!/usr/bin/env bash

# Activate a self-contained QM9 runtime copied to the current compute node.
# The root may retain its historical node02 path name for immutable protocol
# compatibility; execution identity is always recorded from the real hostname.
export QM9_NODE02_ROOT="${QM9_NODE02_ROOT:-/home/shenwei01/xzh_node02_20260724}"
export QM9_CODE_ROOT="${QM9_CODE_ROOT:-${QM9_NODE02_ROOT}/work/structures25}"
export QM9_LOCAL_RUNTIME="${QM9_LOCAL_RUNTIME:-${QM9_NODE02_ROOT}/runtime_parent/_runtime}"

if [[ ! -x "${QM9_CODE_ROOT}/.venv/bin/python" ]]; then
  echo "ERROR: node-local QM9 Python is missing: ${QM9_CODE_ROOT}/.venv/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -d "${QM9_LOCAL_RUNTIME}/qm9_p1" ]]; then
  echo "ERROR: node-local QM9 dataset is missing: ${QM9_LOCAL_RUNTIME}/qm9_p1" >&2
  return 1 2>/dev/null || exit 1
fi

export DFT_DATA="${DFT_DATA:-${QM9_LOCAL_RUNTIME}/qm9_p1}"
export DFT_MODELS="${DFT_MODELS:-${QM9_NODE02_ROOT}/models}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${QM9_NODE02_ROOT}/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${QM9_NODE02_ROOT}/python}"
export TMPDIR="${TMPDIR:-${QM9_NODE02_ROOT}/tmp}"
export HF_HOME="${HF_HOME:-${QM9_NODE02_ROOT}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${QM9_NODE02_ROOT}/cache/torch}"
export PYTHON_BIN="${PYTHON_BIN:-${QM9_CODE_ROOT}/.venv/bin/python}"
export TORCHRUN="${TORCHRUN:-${QM9_CODE_ROOT}/.venv/bin/torchrun}"
export PATH="${QM9_CODE_ROOT}/.venv/bin:${QM9_NODE02_ROOT}/tools:${PATH}"

mkdir -p "${DFT_MODELS}" "${TMPDIR}" "${HF_HOME}" "${TORCH_HOME}"

echo "QM9_EXECUTION_HOST=$(hostname -s)"
echo "QM9_CODE_ROOT=${QM9_CODE_ROOT}"
echo "QM9_LOCAL_RUNTIME=${QM9_LOCAL_RUNTIME}"
echo "DFT_DATA=${DFT_DATA}"
echo "DFT_MODELS=${DFT_MODELS}"
echo "PYTHON_BIN=${PYTHON_BIN}"
