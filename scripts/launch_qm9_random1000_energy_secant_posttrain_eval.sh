#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

BASELINE_RUN="${BASELINE_RUN:-${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203}"
BASELINE_CKPT="${BASELINE_CKPT:-${BASELINE_RUN}/checkpoints/epoch_009.ckpt}"
CANDIDATE_RUN="${CANDIDATE_RUN:-${DFT_MODELS}/train/runs/qm9_random1000_energy_secant_w0p1_e10_20260715_171103}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-${CANDIDATE_RUN}/checkpoints/last.ckpt}"
REFERENCE_DIR="${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians"
MOLECULES="0000777,0040728,0000242,0003027,0021889,0132081,0000835,0004063,0020643,0011449"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_energy_secant_posttrain/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}/force_proxy" "${OUT_DIR}/fixed_density_incomplete_proxy" "${OUT_DIR}/logs"

for path in "${BASELINE_CKPT}" "${CANDIDATE_CKPT}" "${REFERENCE_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required input: ${path}" >&2
    exit 1
  fi
done

names=(Baseline Candidate Baseline Candidate)
runs=("${BASELINE_RUN}" "${CANDIDATE_RUN}" "${BASELINE_RUN}" "${CANDIDATE_RUN}")
ckpts=("${BASELINE_CKPT}" "${CANDIDATE_CKPT}" "${BASELINE_CKPT}" "${CANDIDATE_CKPT}")
splits=(val val test test)
for index in 0 1 2 3; do
  name="${names[$index]}"
  split="${splits[$index]}"
  (
    export CUDA_VISIBLE_DEVICES="${index}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/force_${split}_${name}.time.txt" \
      python scripts/qm9_force_eval.py \
        --run-dir "${runs[$index]}" \
        --ckpt "${ckpts[$index]}" \
        --output-json "${OUT_DIR}/force_proxy/${split}_${name}.json" \
        --worst-csv "${OUT_DIR}/force_proxy/${split}_${name}_worst.csv" \
        --split "${split}" --ground-state-only \
        --device cuda:0 --batch-size 4 --num-workers 4 --top-k 30 \
      >"${OUT_DIR}/logs/force_${split}_${name}.log" 2>&1
  )
done

export CUDA_VISIBLE_DEVICES=4
/usr/bin/time -v -o "${OUT_DIR}/logs/fixed_density_incomplete_proxy.time.txt" \
  python scripts/qm9_second_order_autograd_hessian_audit.py \
    --run "Baseline=${BASELINE_RUN}=${BASELINE_CKPT}" \
    --run "Candidate=${CANDIDATE_RUN}=${CANDIDATE_CKPT}" \
    --molecules "${MOLECULES}" --split test \
    --scf-iteration -1 --num-workers 0 --device cuda:0 --model-dtype float64 \
    --output-json "${OUT_DIR}/fixed_density_incomplete_proxy/summary.json" \
    --output-dir "${OUT_DIR}/fixed_density_incomplete_proxy" \
    --reference-dir "${REFERENCE_DIR}" \
    --fd-displacement 1e-3 --hvp-eps 1e-3 \
    --directional-sample-ids 1 2 3 --compare-pbe-force-secant \
    --run-hvp --no-run-self-edge-diagnostic --no-run-unrolled \
    >"${OUT_DIR}/logs/fixed_density_incomplete_proxy.log" 2>&1

python - "${OUT_DIR}" "${BASELINE_CKPT}" "${CANDIDATE_CKPT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {
    "definition": (
        "Fast regression funnel only. force_proxy and fixed_density_incomplete_proxy "
        "differentiate the learned kin_plus_xc scalar and are not complete total-OFDFT forces."
    ),
    "checkpoints": {
        "Baseline": sys.argv[2],
        "CandidateFinal": sys.argv[3],
    },
    "force_proxy": {},
    "fixed_density_incomplete_proxy": str(
        out / "fixed_density_incomplete_proxy" / "summary.json"
    ),
}
for split in ("val", "test"):
    summary["force_proxy"][split] = {}
    for name in ("Baseline", "Candidate"):
        payload = json.loads((out / "force_proxy" / f"{split}_{name}.json").read_text())
        summary["force_proxy"][split][name] = {
            key: payload.get(key)
            for key in (
                "energy_mae",
                "energy_rmse",
                "force_component_mae",
                "force_component_rmse",
                "force_vector_mae",
                "samples_evaluated",
            )
        }
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "posttrain_fast_eval_dir=${OUT_DIR}"
