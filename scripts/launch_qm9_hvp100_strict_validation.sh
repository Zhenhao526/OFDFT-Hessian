#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?run name required}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}" DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATA_ROOT="${HVP_DATA_ROOT:-${DFT_MODELS}/hvp100/20260716}"
ROOT="${HVP_RESULT_ROOT:-${DFT_MODELS}/hvp100/20260716_gated25v2}"
RUN_DIR="${DFT_MODELS}/train/runs/${RUN_NAME}"
CKPT="${RUN_DIR}/checkpoints/last.ckpt"
OUT="${ROOT}/strict_validation/${RUN_NAME}"
MOLECULES="$(python - <<'PY'
import json
from pathlib import Path
p=Path('/scratch/xzh/models/eval/qm9_random1000_validation_protocol/validation_representatives_20.json')
print(','.join(row['molecule_id'] for row in json.loads(p.read_text())))
PY
)"
mkdir -p "${OUT}"

/usr/bin/time -v -o "${OUT}/resource.time" \
python scripts/qm9_total_ofdft_hvp_audit.py \
  --dataset-dir "${DFT_DATA}/QM9PBEForceRandom1000" \
  --reference-dir "${DATA_ROOT}/pbe_hessians/cache" \
  --direction-sidecar-dir "${DATA_ROOT}/hvp_sidecars/sidecars" \
  --run "${RUN_NAME}=${RUN_DIR}=${CKPT}" \
  --molecules "${MOLECULES}" --sample-id 0 --direction-coordinate 0 \
  --hvp-step 1e-5 --mixed-derivative-step 1e-4 \
  --integral-derivative-step 1e-4 --integral-derivative-workers 4 \
  --model-geometry-derivative autograd \
  --response-solver auto --krylov-tolerance 1e-8 \
  --max-krylov-iterations 1200 --preconditioner-probes 8 \
  --dense-fallback-max-coefficients 4096 \
  --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
  --fallback-optimizer adam --fallback-lr 3e-4 \
  --fallback-max-cycle 10000 --fallback-convergence-tolerance 1e-5 \
  --fallback-always --lbfgs-refine --newton-refine \
  --device cuda:0 --transform-device cpu --output-dir "${OUT}"
