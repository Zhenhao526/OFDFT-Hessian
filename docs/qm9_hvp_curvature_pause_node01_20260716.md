# QM9 HVP Curvature Pause Snapshot

Status: paused by user request on 2026-07-16 17:40 Asia/Singapore.

Resume event: the user explicitly resumed the pipeline on 2026-07-16 17:46:53 Asia/Singapore.
The immutable pause inventory remains the resume boundary. Node01-only jobs are train100 missing
tasks `2330`, validation remaining12 `2331`, val20 postprocessing `2332`, and train100
postprocessing/A--E orchestration `2333`. No other node is permitted by these submissions.

## Pause state

All current HVP branch-audit jobs and their downstream dependencies were cancelled cleanly:

- train100 arrays: Slurm `1443`, `1464`;
- validation remaining12: Slurm `1629`;
- val20 postprocessing: Slurm `1797`;
- train100 postprocessing and A--E launch: Slurm `1865`.

The user queue is empty. A--E screen/multiseed training never started, so there are no new
curvature-model checkpoints to checkpoint or merge. Existing source labels, checkpoints, PBE
references and completed branch-audit artifacts were not modified or deleted. Test100 remains
unread for this experiment.

## Frozen inventory

Remote archive root:

```text
/scratch/xzh/models/hvp_curvature_v1/20260716/pause_node01_20260716
```

An identical local mirror is stored at
`_runtime/qm9_hvp_curvature_pause_node01_20260716`; all five hashes match the remote archive.

| item | value |
|---|---:|
| expected train100 tasks | 1200 |
| validated complete tasks | 830 |
| missing tasks | 370 |
| missing tasks with partial directories | 40 |
| completed NPZ bytes | 7,004,919,300 |
| validation errors | 0 |

| archive file | SHA256 |
|---|---|
| `pause_snapshot.json` | `10d34562e18dcd634816e83c7ee3b81dbcce65715c86e42a2d4fac9248357199` |
| `completed_train100_tasks.tsv` | `7fe576f39ab480bb93054654b5577a66f7f503eceb9f0cde71250919de24c90c` |
| `missing_train100_tasks.tsv` | `adbff87bf477459efcc4d8f3a333d6cb05317d3dd504965f45eea6379d6f90aa` |
| `partial_train100_tasks.tsv` | `5af5ff97f4991d5e339df233b3e9225f2f54baa2453fd8903ca6b14f3366a6e3` |
| `completed_artifacts.tsv` | `512e0ad23a2900b28f97bff5aede5b6603614913cbfe15e03db614a1b02cf88a` |

The snapshot also freezes hashes for the source split, train100 IDs, multidirection manifest,
baseline epoch-9 checkpoint, screen table and all four active audit/training protocols. The
baseline checkpoint hash is `722afe50...f1f96`; the train task table remains
`bc658ab5...13dacd`.

## Node policy

Unless the user explicitly names another node, all future computation for this project must use
only `node01`. The active HVP pipeline Slurm scripts now default to `#SBATCH --nodelist=node01`.
GPU arrays must use at most 8 concurrent tasks on this node. Other nodes require a new explicit
instruction.

## Resume procedure

The following command was run once after the user explicitly requested resumption:

```bash
bash /scratch/xzh/code/structures25/scripts/resume_qm9_hvp_curvature_node01.sh
```

The script submits only the frozen 370-task missing table, including recomputation of the 40
partial tasks, at `0-7` concurrency on node01. It then runs the untouched validation remaining12,
train/val postprocessing and A--E validation funnel on node01. Completed 830 tasks are not
resubmitted. The resume script does not delete partial directories and does not access Test100.

## Code support

- `scripts/archive_qm9_hvp_curvature_pause.py`: reproducible inventory/hash generator;
- `scripts/resume_qm9_hvp_curvature_node01.sh`: non-automatic node01-only resume entry point;
- `scripts/slurm_qm9_hvp_train100_postprocess.sbatch`: now requires the active val-post job ID,
  preventing accidental reuse of cancelled job `1797`.
