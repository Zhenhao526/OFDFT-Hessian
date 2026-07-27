#!/usr/bin/env bash
set -euo pipefail

node=${1:-node05}
workspace=/home/shenwei01/WT_Al_melting_workspace_20260724
runtime=/home/shenwei01/wt_melting_runtime_20260724
repository=$workspace/repository
source_root=$workspace/runs/xwm_lkt_20260727/xwm/volume_confirmation_T0975_steps300_r1
staged_source=$workspace/xwm_lkt_seeds/xwm_T0975_volume_confirmed_r1
run_root=$workspace/runs/xwm_lkt_20260727/xwm/volume_confirmation_T0900_steps300_r1
config=$repository/config/abacus_xwm_cpu36.json

ssh "$node" "test ! -e '$run_root'"
ssh "$node" "mkdir -p '$workspace'"
rsync -a "$runtime/" "$node:$runtime/"
rsync -a "$repository/" "$node:$repository/"
rsync -a "$workspace/assets/" "$node:$workspace/assets/"
ssh "$node" "mkdir -p '$staged_source'"
rsync -a "$source_root/solid/" "$node:$staged_source/solid/"
rsync -a "$source_root/liquid/" "$node:$staged_source/liquid/"

ssh "$node" "
  cd '$repository'
  env PYTHONPATH=. python3 scripts/prepare_kedf_volume_confirmation.py prepare \
    --out '$run_root' \
    --solid-source '$staged_source/solid' \
    --liquid-source '$staged_source/liquid' \
    --solid-volume 17.93 \
    --liquid-volume 18.92 \
    --config '$config' \
    --temperature 900 \
    --steps 300 \
    --csvr-tau 5 \
    --seed 2026072790 \
    --ranks 36
  tmux new-session -d -s xwm_T900_vconf_r1 \
    \"SOLID_CPUS=40-75 LIQUID_CPUS=76-111 bash '$repository/remote_staging/run_kedf_volume_confirmation_node_local.sh' '$run_root' '$repository' >'$run_root/pipeline.log' 2>&1\"
  tmux has-session -t xwm_T900_vconf_r1
"
