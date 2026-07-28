#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${DFT_MODELS}/eval/qm9_random1000_hessian_protocol_scan/20260714_194632/representative_manifest_8mol.json}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_same_gpu_concurrency/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}/baseline" "${OUT_DIR}/concurrent/a" "${OUT_DIR}/concurrent/b" "${OUT_DIR}/logs"

"${PYTHON_BIN}" - "${SOURCE_MANIFEST}" "${OUT_DIR}" <<'PY'
import json,sys
from pathlib import Path
source,out=Path(sys.argv[1]),Path(sys.argv[2])
by_id={row['molecule_id']:row for row in json.load(open(source))}
ids=['0000777','0043905','0040728','0060531']
rows=[by_id[x] for x in ids]
(out/'manifest_all.json').write_text(json.dumps(rows,indent=2)+'\n')
# Balance approximate coordinate work: 7+27 atoms versus 14+21 atoms.
(out/'manifest_a.json').write_text(json.dumps([rows[0],rows[3]],indent=2)+'\n')
(out/'manifest_b.json').write_text(json.dumps([rows[1],rows[2]],indent=2)+'\n')
PY

common=(
  --dataset-dir "${DATASET_DIR}"
  --run "EGF_w0p1=${EGF_RUN_DIR}=${EGF_CKPT}"
  --displacement 1e-3 --initialization sad_default --base-density-warm-start
  --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2
  --fallback-optimizer adam --fallback-lr 3e-4 --fallback-max-cycle 10000
  --fallback-convergence-tolerance 1e-4 --fallback-always --device cuda:0
)

run_eval() {
  local gpu="$1" name="$2" manifest="$3" count="$4" root="$5"
  mkdir -p "${root}/hessians"
  local start
  start="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/qm9_hessian_density_relaxed_eval.py \
    --manifest-json "${manifest}" "${common[@]}" --max-molecules "${count}" \
    --output-json "${root}/summary.json" --output-csv "${root}/molecules.csv" \
    --optimization-csv "${root}/optimizations.csv" --hessian-npz-dir "${root}/hessians" \
    >"${OUT_DIR}/logs/${name}.log" 2>&1
  echo "$(($(date +%s)-start))" > "${root}/wall_seconds.txt"
}

run_eval 0 baseline "${OUT_DIR}/manifest_all.json" 4 "${OUT_DIR}/baseline" &
baseline_pid=$!
(
  concurrent_start="$(date +%s)"
  run_eval 1 concurrent_a "${OUT_DIR}/manifest_a.json" 2 "${OUT_DIR}/concurrent/a" &
  pid_a=$!
  run_eval 1 concurrent_b "${OUT_DIR}/manifest_b.json" 2 "${OUT_DIR}/concurrent/b" &
  pid_b=$!
  wait "${pid_a}"
  wait "${pid_b}"
  echo "$(($(date +%s)-concurrent_start))" > "${OUT_DIR}/concurrent/wall_seconds.txt"
) &
concurrent_pid=$!
wait "${baseline_pid}"
wait "${concurrent_pid}"

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
baseline=json.load(open(root/'baseline/summary.json'))
parts=[json.load(open(root/'concurrent/a/summary.json')),json.load(open(root/'concurrent/b/summary.json'))]
base_rows={row['molecule_id']:row for row in baseline['rows'] if row.get('success')}
part_rows={row['molecule_id']:row for part in parts for row in part['rows'] if row.get('success')}
comparisons=[]
for molecule_id,row in sorted(base_rows.items()):
    other=part_rows[molecule_id]
    hb=np.load(row['hessian_npz'])['density_relaxed_hessian']
    hc=np.load(other['hessian_npz'])['density_relaxed_hessian']
    comparisons.append({
      'molecule_id':molecule_id,
      'relative_fro_difference':float(np.linalg.norm(hb-hc)/np.linalg.norm(hb)),
      'mae_difference':float(abs(row['mae']-other['mae'])),
    })
baseline_wall=int((root/'baseline/wall_seconds.txt').read_text())
concurrent_wall=int((root/'concurrent/wall_seconds.txt').read_text())
result={
  'definition':'equal four-molecule workload: one worker on GPU0 versus two independent workers sharing GPU1',
  'baseline_wall_s':baseline_wall,'two_workers_same_gpu_wall_s':concurrent_wall,
  'speedup':baseline_wall/concurrent_wall,
  'baseline_success':len(base_rows),'concurrent_success':len(part_rows),
  'max_relative_fro_difference':max(x['relative_fro_difference'] for x in comparisons),
  'comparisons':comparisons,
}
(root/'analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2,sort_keys=True))
PY

echo "Same-GPU concurrency calibration complete: ${OUT_DIR}"
