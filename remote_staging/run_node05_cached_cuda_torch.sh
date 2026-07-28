#!/usr/bin/env bash
set -euo pipefail

cache_root="${HOME}/.cache/uv/archive-v0"
python_bin="${HOME}/WT_Al_melting_workspace_20260724/.venv-reference-cuda/bin/python"

python_roots=(
  0MCM-e98B3WHIuhr
  1ufxLt9Prpqp0QHz
  rgWtqNmEDlIarMt6
  8mKzA8sMs89lGViz
  9TqT9tGMZOL-IUgm
  3xEz7GPBoKyqeoSy
  CAcORDrSVdLQU4XU
  Pf4hVQEQzIUGovwn
  ciW-6YG4IhsTOiT0
  leWE-KeqsX--r9Hf
  mg99DSySV6XPkNvS
  TCAN_VaqJzKbaRjR
  3uKrdTdDrVHQ9e7f
  rqEPnAkXhBUt5b6B
  TyNDrWAzOhOPsTrL
  9pKYWyVV_LyUEroU
  7I9cgv5X8BAr_M_r
  QlQ1VpObBebBm72J
  X7LjIlkFLcvBVRmP
  vr0OwL7OrIUOxDX-
  BMa_wHohjjxS-q9z
  WQ6ptA-2QGrY3Am_
  UC0eEE2JdXVQLyKb
  2DDedYjDYK8O2QLH
  1nDfO9ynMV4sporO
)

library_roots=(
  0MCM-e98B3WHIuhr/torch/lib
  TCAN_VaqJzKbaRjR/nvidia/cuda_nvrtc/lib
  3uKrdTdDrVHQ9e7f/nvidia/cuda_runtime/lib
  rqEPnAkXhBUt5b6B/nvidia/cuda_cupti/lib
  TyNDrWAzOhOPsTrL/nvidia/cudnn/lib
  9pKYWyVV_LyUEroU/nvidia/cublas/lib
  7I9cgv5X8BAr_M_r/nvidia/cufft/lib
  QlQ1VpObBebBm72J/nvidia/curand/lib
  X7LjIlkFLcvBVRmP/nvidia/cusolver/lib
  vr0OwL7OrIUOxDX-/nvidia/cusparse/lib
  BMa_wHohjjxS-q9z/nvidia/cusparselt/lib
  WQ6ptA-2QGrY3Am_/nvidia/nccl/lib
  UC0eEE2JdXVQLyKb/nvidia/nvtx/lib
  2DDedYjDYK8O2QLH/nvidia/nvjitlink/lib
  1nDfO9ynMV4sporO/nvidia/cufile/lib
)

join_cache_roots() {
  local joined=""
  local root
  for root in "$@"; do
    if [[ ! -e "${cache_root}/${root}" ]]; then
      echo "missing cached dependency: ${cache_root}/${root}" >&2
      return 2
    fi
    if [[ -n "${joined}" ]]; then
      joined+=":"
    fi
    joined+="${cache_root}/${root}"
  done
  printf '%s' "${joined}"
}

[[ -x "${python_bin}" ]] || {
  echo "missing isolated Python interpreter: ${python_bin}" >&2
  exit 2
}

export PYTHONPATH="$(join_cache_roots "${python_roots[@]}")${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="$(join_cache_roots "${library_roots[@]}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ $# -eq 0 ]]; then
  exec "${python_bin}" -c \
    'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
fi

exec "${python_bin}" "$@"
