#!/usr/bin/env bash
#SBATCH --job-name=gf_cth_dirs
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/graphformer_relaxed_hvp/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/graphformer_relaxed_hvp/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

parent_set=${PARENT_SET:?set PARENT_SET=stable5 or train20}
root=/scratch/xzh/models/graphformer_relaxed_hvp/20260723/v1
output="${root}/directions/${parent_set}"
mkdir -p "${output}" /scratch/xzh/logs/graphformer_relaxed_hvp
/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/prepare_qm9_graphformer_relaxed_hvp_directions.py \
    --protocol configs/audit/qm9_graphformer_complete_total_relaxed_hvp_v1.yaml \
    --parent-set "${parent_set}" \
    --output-dir "${output}"
