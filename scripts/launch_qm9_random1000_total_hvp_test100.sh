#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
REFERENCE_DIR="${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians"
REFERENCE_MANIFEST="${REFERENCE_MANIFEST:-${DFT_MODELS}/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/20260714_135600/pbe_hessian_manifest_test100_gpu4pyscf.json}"
BASELINE_RUN="${BASELINE_RUN:-${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203}"
BASELINE_CKPT="${BASELINE_CKPT:-${BASELINE_RUN}/checkpoints/epoch_009.ckpt}"
CANDIDATE_RUN="${CANDIDATE_RUN:?CANDIDATE_RUN is required}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:?CANDIDATE_CKPT is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
N_SHARDS="${N_SHARDS:-8}"
mkdir -p "${OUT_DIR}/workers" "${OUT_DIR}/logs" "${OUT_DIR}/shards"

python - "${REFERENCE_MANIFEST}" "${OUT_DIR}/shards" "${N_SHARDS}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out = Path(sys.argv[2])
n_shards = int(sys.argv[3])
rows = [row for row in json.loads(manifest.read_text()) if row.get("success")]
if len(rows) != 100:
    raise RuntimeError(f"Expected 100 successful Test100 references, found {len(rows)}")
shards = [[] for _ in range(n_shards)]
loads = [0 for _ in range(n_shards)]
for row in sorted(rows, key=lambda item: (-int(item["natoms"]), item["molecule_id"])):
    index = min(range(n_shards), key=lambda value: loads[value])
    shards[index].append(row)
    loads[index] += int(row["natoms"]) ** 3
for index, shard in enumerate(shards):
    (out / f"shard_{index:02d}.txt").write_text(
        "\n".join(row["molecule_id"] for row in shard) + "\n"
    )
    (out / f"shard_{index:02d}.json").write_text(json.dumps(shard, indent=2) + "\n")
print(json.dumps({"molecules": len(rows), "loads": loads, "sizes": list(map(len, shards))}))
PY

run_worker() {
  local model="$1" shard="$2" gpu="$3" run_dir="$4" checkpoint="$5"
  local shard_tag
  shard_tag="$(printf '%02d' "${shard}")"
  local worker_dir="${OUT_DIR}/workers/${model}_shard${shard_tag}"
  local molecules
  molecules="$(paste -sd, "${OUT_DIR}/shards/shard_${shard_tag}.txt")"
  mkdir -p "${worker_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/${model}_shard${shard_tag}.time.txt" \
      python scripts/qm9_total_ofdft_hvp_audit.py \
        --dataset-dir "${DATASET_DIR}" --reference-dir "${REFERENCE_DIR}" \
        --run "${model}=${run_dir}=${checkpoint}" \
        --molecules "${molecules}" --sample-id 0 --direction-coordinate 0 \
        --hvp-step 1e-5 --mixed-derivative-step 1e-4 \
        --integral-derivative-step 1e-4 --integral-derivative-workers 2 \
        --model-geometry-derivative autograd \
        --response-solver auto --krylov-tolerance 1e-8 \
        --max-krylov-iterations 1200 --preconditioner-probes 8 \
        --dense-fallback-max-coefficients 4096 \
        --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
        --fallback-optimizer adam --fallback-lr 3e-4 \
        --fallback-max-cycle 10000 --fallback-convergence-tolerance 1e-5 \
        --fallback-always --lbfgs-refine --newton-refine \
        --device cuda:0 --transform-device cpu --output-dir "${worker_dir}" \
        >"${OUT_DIR}/logs/${model}_shard${shard_tag}.log" 2>&1
  ) &
  pids+=("$!")
}

pids=()
for shard in $(seq 0 $((N_SHARDS - 1))); do
  run_worker Baseline "${shard}" "${shard}" "${BASELINE_RUN}" "${BASELINE_CKPT}"
  sleep 2
  run_worker Candidate "${shard}" "${shard}" "${CANDIDATE_RUN}" "${CANDIDATE_CKPT}"
  sleep 2
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  echo "ERROR: ${failures} HVP workers failed" >&2
  exit 1
fi

python - "${OUT_DIR}" "${N_SHARDS}" "${BASELINE_CKPT}" "${CANDIDATE_CKPT}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
n_shards = int(sys.argv[2])
checkpoints = {"Baseline": sys.argv[3], "Candidate": sys.argv[4]}
all_rows = []
for model in ("Baseline", "Candidate"):
    for shard in range(n_shards):
        path = out / "workers" / f"{model}_shard{shard:02d}" / "summary.json"
        payload = json.loads(path.read_text())
        if len(payload["summaries"]) == 0:
            raise RuntimeError(f"Empty worker output: {path}")
        all_rows.extend(payload["summaries"])
if len(all_rows) != 200:
    raise RuntimeError(f"Expected 200 model/molecule rows, found {len(all_rows)}")

flat_rows = []
for row in all_rows:
    flat_rows.append(
        {
            "run": row["run"],
            "molecule_id": row["molecule_id"],
            "natoms": row["natoms"],
            "strict_vs_pbe_mae": row["strict_relaxed_vs_pbe"]["mae"],
            "strict_vs_pbe_rmse": row["strict_relaxed_vs_pbe"]["rmse"],
            "strict_vs_pbe_relative_frobenius": row["strict_relaxed_vs_pbe"][
                "relative_frobenius"
            ],
            "implicit_vs_strict_relative_frobenius": row["implicit_vs_relaxed"][
                "relative_frobenius"
            ],
            "response_solver": row["response_solver"],
            "response_relative_residual": row["response_relative_residual"],
            "max_density_gradient": row["strict_relaxed_max_gradient_norm"],
            "wall_time_s": row["wall_time_s"],
        }
    )
fields = sorted({key for row in flat_rows for key in row})
with (out / "per_molecule.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(sorted(flat_rows, key=lambda row: (row["molecule_id"], row["run"])))

by_model = {
    model: [row for row in flat_rows if row["run"] == model]
    for model in ("Baseline", "Candidate")
}
summaries = {}
for model, rows in by_model.items():
    summaries[model] = {"molecules": len(rows)}
    for key in (
        "strict_vs_pbe_mae",
        "strict_vs_pbe_rmse",
        "strict_vs_pbe_relative_frobenius",
        "implicit_vs_strict_relative_frobenius",
        "wall_time_s",
    ):
        values = [float(row[key]) for row in rows]
        summaries[model][f"mean_{key}"] = sum(values) / len(values)
        summaries[model][f"median_{key}"] = statistics.median(values)

baseline = {row["molecule_id"]: row for row in by_model["Baseline"]}
candidate = {row["molecule_id"]: row for row in by_model["Candidate"]}
wins = sum(
    candidate[molecule]["strict_vs_pbe_mae"] < row["strict_vs_pbe_mae"]
    for molecule, row in baseline.items()
)
result = {
    "definition": (
        "Frozen Test100 coordinate-0 HVP from strict complete scalar-derived total-OFDFT "
        "force differences; implicit KKT response is a cross-check only."
    ),
    "checkpoints": checkpoints,
    "summaries": summaries,
    "candidate_mae_wins": wins,
    "candidate_mae_losses": 100 - wins,
    "per_molecule_csv": str(out / "per_molecule.csv"),
}
(out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

echo "total_hvp_test100_dir=${OUT_DIR}"
