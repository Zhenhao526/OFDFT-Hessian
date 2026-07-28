#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SWEEP_STAMP="${SWEEP_STAMP:-20260714_202203}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_lambda_tier1/${RUN_STAMP}}"
REFERENCE_DIR="${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians"
mkdir -p "${OUT_DIR}/force" "${OUT_DIR}/fixed_hessian" "${OUT_DIR}/logs"

python scripts/select_qm9_split_representatives.py \
  --split-file "${DFT_DATA}/QM9PBEForceRandom1000/split.pkl" \
  --data-dir "${DFT_DATA}" \
  --split val --count 20 \
  --output-json "${OUT_DIR}/validation_representatives_20.json" \
  --output-ids "${OUT_DIR}/validation_representatives_20.ids" \
  >"${OUT_DIR}/logs/select_validation_representatives.log" 2>&1
MOLECULES="$(cat "${OUT_DIR}/validation_representatives_20.ids")"

names=(EG EGF_w0p1 EGF_w0p3 EGF_w1p0 EGF_w3p0 EGF_w10p0)
runs=(
  "${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829"
  "${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829"
  "${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew0p3_e10_${SWEEP_STAMP}"
  "${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew1p0_e10_${SWEEP_STAMP}"
  "${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew3p0_e10_${SWEEP_STAMP}"
  "${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew10p0_e10_${SWEEP_STAMP}"
)

run_args=()
pids=()
for idx in "${!names[@]}"; do
  name="${names[$idx]}"
  run_dir="${runs[$idx]}"
  ckpt="${run_dir}/checkpoints/epoch_009.ckpt"
  if [[ ! -f "${ckpt}" && -f "${run_dir}/checkpoints/last.ckpt" ]]; then
    ckpt="${run_dir}/checkpoints/last.ckpt"
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "ERROR: missing checkpoint ${ckpt}" >&2
    exit 1
  fi
  run_args+=(--run "${name}=${run_dir}=${ckpt}")
  (
    export CUDA_VISIBLE_DEVICES="${idx}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/force_${name}.time.txt" \
      python scripts/qm9_force_eval.py \
        --run-dir "${run_dir}" \
        --ckpt "${ckpt}" \
        --output-json "${OUT_DIR}/force/${name}.json" \
        --worst-csv "${OUT_DIR}/force/${name}_worst.csv" \
        --split val --ground-state-only \
        --device cuda:0 --batch-size 4 --num-workers 0 --top-k 30 \
        >"${OUT_DIR}/logs/force_${name}.log" 2>&1
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

export CUDA_VISIBLE_DEVICES=6
/usr/bin/time -v -o "${OUT_DIR}/logs/fixed_hessian.time.txt" \
  python scripts/qm9_second_order_autograd_hessian_audit.py \
    "${run_args[@]}" \
    --molecules "${MOLECULES}" --split val \
    --scf-iteration -1 --num-workers 0 --device cuda:0 --model-dtype float64 \
    --output-json "${OUT_DIR}/fixed_hessian/summary.json" \
    --output-dir "${OUT_DIR}/fixed_hessian" \
    --reference-dir "${REFERENCE_DIR}/not_used_for_validation" \
    --fd-displacement 1e-3 --hvp-eps 1e-3 \
    --directional-sample-ids 1 2 3 --compare-pbe-force-secant \
    --run-hvp --no-run-self-edge-diagnostic --no-run-unrolled \
    >"${OUT_DIR}/logs/fixed_hessian.log" 2>&1

python scripts/qm9_lambda_sweep_tier1_analysis.py \
  --force-dir "${OUT_DIR}/force" \
  --hessian-summary "${OUT_DIR}/fixed_hessian/summary.json" \
  --output-dir "${OUT_DIR}/analysis" \
  >"${OUT_DIR}/logs/analysis.log" 2>&1
echo "tier1 screening complete: ${OUT_DIR}"
