#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 FORMAL_ROOT REPOSITORY PAIR_DAT" >&2
  exit 2
fi

formal_root=$1
repository=$2
pair_dat=$3
log=$formal_root/formal_pipeline.log
cpu_ranges=(0-11 12-23 24-35 38-49 50-61 62-73)

[[ -f "$formal_root/formal_manifest.json" ]]
[[ -f "$pair_dat" ]]
for phase in solid liquid; do
  [[ -f "$formal_root/$phase/manifest.json" ]]
done

mapfile -t points < <(
  python3 - "$formal_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for phase in ("solid", "liquid"):
    manifest = json.loads((root / phase / "manifest.json").read_text())
    for window in manifest["windows"]:
        print(root / phase / window["label"])
PY
)
mapfile -t lambdas < <(
  python3 - "$formal_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for phase in ("solid", "liquid"):
    manifest = json.loads((root / phase / "manifest.json").read_text())
    for window in manifest["windows"]:
        print(window["lambda"])
PY
)

if [[ ${#points[@]} -ne 18 || ${#lambdas[@]} -ne 18 ]]; then
  echo "expected 18 formal phase/lambda windows" >&2
  exit 2
fi
for point in "${points[@]}"; do
  [[ -x "$point/run_local.sh" ]]
  if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
    echo "refusing existing output below $point" >&2
    exit 2
  fi
done

run_wave() {
  local first=$1
  local count=$2
  local pids=()
  local index point lambda cpus status

  for ((slot=0; slot<count; slot++)); do
    index=$((first + slot))
    point=${points[$index]}
    lambda=${lambdas[$index]}
    cpus=${cpu_ranges[$slot]}
    (
      cd "$point"
      exec taskset -c "$cpus" env \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        MPN_TI_LAMBDA="$lambda" MPN_TI_PAIR_MODEL="$pair_dat" \
        ./run_local.sh
    ) >"$point/run.stdout" 2>&1 &
    pids+=("$!")
    printf '%s started point=%s lambda=%s cpus=%s pid=%s\n' \
      "$(date -Iseconds)" "$point" "$lambda" "$cpus" "$!" >>"$log"
  done

  for ((slot=0; slot<count; slot++)); do
    index=$((first + slot))
    if wait "${pids[$slot]}"; then
      printf '%s finished point=%s\n' \
        "$(date -Iseconds)" "${points[$index]}" >>"$log"
    else
      status=$?
      printf '%s failed point=%s status=%s\n' \
        "$(date -Iseconds)" "${points[$index]}" "$status" >>"$log"
      touch "$formal_root/formal_pipeline.failed"
      return 1
    fi
  done
}

: >"$log"
run_wave 0 6
run_wave 6 6
run_wave 12 6

cd "$repository"
for phase in solid liquid; do
  reports=()
  for suffix in d25 d50 d75; do
    case $suffix in
      d25) discard=0.25 ;;
      d50) discard=0.50 ;;
      d75) discard=0.75 ;;
    esac
    report=$formal_root/$phase/ti_prod_${suffix}.json
    env PYTHONPATH=. python3 scripts/analyze_wt_pair_ti_production.py \
      "$formal_root/$phase" \
      --discard-fraction "$discard" \
      --temperature-tolerance 20 \
      --max-block-se 1 \
      --max-half-drift 2 \
      --max-quadrature-difference 2 \
      --minimum-overlap-ess 0.05 \
      --max-overlap-closure 2 \
      --out "$report" >>"$log" 2>&1
    reports+=("$report")
  done
  env PYTHONPATH=. python3 scripts/summarize_wt_ti_convergence.py \
    "${reports[@]}" \
    --max-integral-discard-spread 1 \
    --max-window-discard-spread 1 \
    --max-window-block-se 1 \
    --max-window-half-drift 2 \
    --out "$formal_root/$phase/discard_convergence_summary.json" \
    >>"$log" 2>&1
done

python3 - "$formal_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = {
    phase: json.loads(
        (root / phase / "discard_convergence_summary.json").read_text()
    )
    for phase in ("solid", "liquid")
}
result = {
    "schema": "kedf-ti-formal-production-gate-v1",
    "phase_status": {
        phase: report["status"] for phase, report in summaries.items()
    },
    "status": (
        "verified"
        if all(report["status"] == "verified" for report in summaries.values())
        else "needs_extension_or_refinement"
    ),
}
(root / "formal_gate_summary.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
PY

find "$formal_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name formal_manifest.json \
     -o -name phase_analysis.json -o -name 'ti_prod_*.json' \
     -o -name discard_convergence_summary.json \
     -o -name formal_gate_summary.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$formal_root/SHA256SUMS"

status=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$formal_root/formal_gate_summary.json")
if [[ $status == verified ]]; then
  touch "$formal_root/formal_pipeline.done"
else
  touch "$formal_root/formal_pipeline.needs_extension"
fi
