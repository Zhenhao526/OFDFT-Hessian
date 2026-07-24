#!/usr/bin/env bash
set -euo pipefail

state=/proc/fs/beegfs/c22801-6A1925D1-node01/storage_target_state
workspace=/home/shenwei01/WT_Al_melting_workspace_20260724
source_root=/scratch/xzh/OFDFT-Hessian/runs/al/free_energy_wt
destination=$workspace/staged_inputs/free_energy_wt
log=$workspace/audit/stage_inputs.log
ready=$workspace/audit/staged_inputs.ready

mkdir -p "$workspace/audit" "$destination"
rm -f "$ready"
printf '%s staging_watcher_started\n' "$(date -Iseconds)" >> "$log"

while true; do
    offline=$(grep -c 'Offline' "$state" 2>/dev/null || true)
    printf '%s offline_targets=%s\n' "$(date -Iseconds)" "$offline" >> "$log"
    if [[ $offline -eq 0 ]]; then
        break
    fi
    sleep 300
done

copy_run_tree() {
    local relative=$1
    local source=$source_root/$relative/
    local target=$destination/$relative/

    mkdir -p "$target"
    rsync -a --partial --append-verify --prune-empty-dirs \
        --include='*/' \
        --include='INPUT' \
        --include='STRU' \
        --include='KPT' \
        --include='metadata.json' \
        --include='confirmation_manifest.json' \
        --include='confirmation_summary.json' \
        --include='running_md.log' \
        --include='MD_dump' \
        --include='STRU_MD_*' \
        --exclude='*' \
        "$source" "$target"
}

series=T0900_T0975_T1050_adaptive_T1100_pairv2_steps3000_v2_node01
copy_run_tree fusion_enthalpy_sampling/T0900_continuation_steps3000_v2_node01
copy_run_tree fusion_enthalpy_sampling/$series/critical_extensions/T0975_steps3000_round1
copy_run_tree fusion_enthalpy_sampling/$series/critical_extensions/T0975_steps3000_round2
copy_run_tree fusion_enthalpy_sampling/$series/critical_extensions/T1050_steps3000_round1
copy_run_tree fusion_enthalpy_sampling/$series/critical_extensions/T1050_steps3000_round2

for relative in \
    reference_anchor_audit_T0900_p100sigma1p36_pairv2_v2.json \
    melting_free_energy_T0900_pairv2_v2.json \
    melting_pipeline_pairv2_recovery_T1100_v2_node01/anchor_grid_decision.json \
    fusion_enthalpy_sampling/$series/discard_convergence_summary.json \
    fusion_enthalpy_sampling/$series/discard_convergence_summary_extension_round1.json
do
    mkdir -p "$destination/$(dirname "$relative")"
    rsync -a --partial --append-verify \
        "$source_root/$relative" "$destination/$relative"
done

python3 - "$destination" "$source_root" <<'PY'
import json
import sys
from pathlib import Path

destination = Path(sys.argv[1]).resolve()
source_root = sys.argv[2].rstrip("/")


def relocate(value):
    if isinstance(value, dict):
        return {key: relocate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if isinstance(value, str) and (
        value == source_root or value.startswith(source_root + "/")
    ):
        return str(destination) + value[len(source_root) :]
    return value


for path in destination.rglob("*.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    relocated = relocate(payload)
    path.write_text(
        json.dumps(relocated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
PY

(
    cd "$workspace"
    find staged_inputs -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum
) > "$workspace/audit/staged_inputs.sha256"

{
    printf 'complete_time=%s\n' "$(date -Iseconds)"
    printf 'destination=%s\n' "$destination"
    printf 'file_count=%s\n' "$(find "$destination" -type f | wc -l)"
    printf 'size_bytes=%s\n' "$(du -sb "$destination" | awk '{print $1}')"
    printf 'checksum_manifest=%s\n' "$workspace/audit/staged_inputs.sha256"
} | tee "$ready" >> "$log"
