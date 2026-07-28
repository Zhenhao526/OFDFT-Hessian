#!/usr/bin/env bash
#SBATCH --job-name=ct_3task
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

run_mode=${THREE_TASK_RUN_MODE:-smoke}
protocol=configs/audit/qm9_complete_total_hessian_local_scalar_three_task_pcgrad_v4.yaml
root=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_three_task_pcgrad_v4
arms=(K3_softplus_beta5_three_task K5_tanh_three_task)
arm=${THREE_TASK_ARM_ID:-${arms[${SLURM_ARRAY_TASK_ID}]}}
output=${THREE_TASK_OUTPUT_BASE:-${root}/${run_mode}}/${arm}

python - "${protocol}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

protocol = yaml.safe_load(Path(sys.argv[1]).read_text())
path = Path(protocol["inputs"]["activation_formal_analysis"])
result = json.loads(path.read_text())
required = bool(
    protocol["inputs"]["required_activation_parent_cv_design_authorized"]
)
if bool(result.get("parent_cv_design_authorized")) != required:
    raise SystemExit(
        "three-task contingency is forbidden because activation audit did not fail"
    )
if result.get("test100_accessed") is not False:
    raise SystemExit("activation analysis does not certify frozen Test100")
if int(result.get("test100_evaluations_used", -1)) != 0:
    raise SystemExit("activation analysis Test100 count is not zero")
PY

mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${output}/resource.time" \
python scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py \
  --protocol "${protocol}" \
  --arm-id "${arm}" \
  --run-mode "${run_mode}" \
  --output-dir "${output}" \
  --device cuda:0
