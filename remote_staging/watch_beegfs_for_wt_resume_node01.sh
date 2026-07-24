#!/usr/bin/env bash
set -u

state=/proc/fs/beegfs/c22801-6A1925D1-node01/storage_target_state
log=/home/shenwei01/wt_beegfs_resume_watch.log
ready=/home/shenwei01/wt_beegfs_ready

rm -f "$ready"
printf '%s watcher_started state=%s\n' "$(date -Iseconds)" "$state" >> "$log"

while true; do
    if [[ ! -r $state ]]; then
        printf '%s state_unreadable\n' "$(date -Iseconds)" >> "$log"
        sleep 300
        continue
    fi

    offline=$(grep -c 'Offline' "$state" || true)
    total=$(df -B1 --output=size /scratch 2>/dev/null | tail -1 | tr -d ' ')
    printf '%s offline_targets=%s scratch_total_bytes=%s\n' \
        "$(date -Iseconds)" "$offline" "${total:-unknown}" >> "$log"

    if [[ $offline -eq 0 ]]; then
        printf '%s storage_targets_online scratch_total_bytes=%s\n' \
            "$(date -Iseconds)" "${total:-unknown}" | tee -a "$log" > "$ready"
        exit 0
    fi
    sleep 300
done
