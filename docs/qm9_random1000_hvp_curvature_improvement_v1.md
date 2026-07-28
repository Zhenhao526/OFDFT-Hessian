# QM9 random1000 HVP curvature improvement v1

Status: validation funnel completed on node01, 2026-07-17 12:14:40 Asia/Singapore. No candidate
passed the final frozen gate, and no Test100 structure, label, prediction or metric was used.

## Objective and frozen boundaries

The experiment tests whether curvature supervision can improve validation force and strict
complete-total density-relaxed HVP/Hessian metrics without increasing energy MAE by more than 5%.
Training always retains energy, density-gradient and scalar-energy-derived force losses. There is
no force head. The source split has 800/100/100 parent groups; only the original train800 is used
for optimization, and 100 stratified train parents are candidates for curvature labels.

| artifact | SHA256 |
|---|---|
| parent-grouped source split | `919c5f6c250b0d2e9f14f43522894fcac8b179d6d4fd5b774c2cec1ea249034b` |
| train100 selection | `8982fc6349b520b9b98f52fa684e81558b6afab16d14a100f3c3843608503f0b` |
| four-direction sidecar manifest | `38d0352aad6a50ead75be48ddc33c34112fdd02b5fdc6f9e7f6e5b2ea039ee9b` |
| train100 branch task table | `bc658ab5594f30f4dc9a92080d1f64f9ef61f80477de9afa9a6b34897113dacd` |
| train700 forgetting IDs | `270e30f61fcf763963a70db68ab0715a7f9599f12f117407177cba3113ec827e` |
| A-E screen task table | `61f04a59bdd85dac7d50ae3eba7b1a1c424ffce776b7123388250b4deddce8b5` |

Frozen machine-readable protocols are
`configs/audit/qm9_hvp_branch_stability_v2.yaml`,
`configs/audit/qm9_hvp_branch_stability_v3.yaml`, and
`configs/audit/qm9_hvp_curvature_training_v2.yaml`. The strict validation protocol was frozen,
before any A--E screen metric existed, as
`configs/audit/qm9_hvp_curvature_validation_v1.yaml`.
Before the screen started, training v2 was amended to make the already-computed train700
forgetting metrics hard gates: train700 energy and force MAE may each degrade by at most 5% versus
the matched-seed A run, and all three seeds must pass. Tier-1 fixed-HVP averages were also changed
to use only v3-eligible validation parents and passing directions; unstable samples are reported
but cannot rank hyperparameters. Stage1 evaluation now depends on both screen completion and val20
stability postprocessing. Final pre-screen hashes are training v2
`f6244126567d4e8fff62f78f97496c84b62a20d341b96633ccd5cb2cf9f30d91` and validation v1
`df78b5fb885fb2c47d82677611e1463015559b9e88c8ab3061b83be086352251`.
The 20-row screen TSV remains byte-identical (`61f04a...ce8b5`); no training combination changed.
Both fixed-density ranking and strict complete-total HVP require at least five stable validation
parents. Fewer than five produces diagnostic tables but no candidate or Test100 authorization.

## Derivative stability gate

Each parent has four unit internal directions: random internal, bond stretch, angle bend and a
low-frequency mode. Every direction is audited with SAD-continuation, label-density-continuation
and independent SAD displaced densities. The audit records branch energies, Coulomb-metric density
overlap, complete forces, strict projected density gradients, KKT spectrum/condition number,
implicit response residuals, relaxed-force HVPs at `h={3e-5,1e-5,3e-6} Bohr`, and scalar-energy
closure at `h=1e-3 Bohr`.

V2 requires all four directions to pass. A preregistered train12 pilot gave 19/48 passing
directions but 0/12 all-four parents. Several parents had three stable directions and only a
direction-specific low-frequency finite-difference failure. V3 was frozen at that point, before
the remaining train100 results: density/branch/KKT failures exclude a parent, while response,
step-stability and energy-force-closure failures mask only that direction; at least 3/4 directions
must remain. The same pilot gives 3/12 eligible parents. Numeric thresholds are unchanged and the
V2 result remains mandatory sensitivity evidence.

`0044504` demonstrates why filtering is necessary. Its bond and angle directions pass, while its
random and low-frequency directions have large step/implicit disagreement; it also fails a density
gate and is excluded at parent level rather than averaged into model selection.

Quantitatively, `0044504` does not show a conventional multi-density electronic branch split:
three-branch energy spread is `2.27e-13 Ha`, maximum Coulomb density distance is `6.02e-10`,
maximum branch-force RMSE is `1.18e-9 Ha/Bohr`, KKT condition number is `1.58e7`, and response
relative residual is at most `1.37e-13`. Instead, random/low-frequency directions have HVP-step
instability `8.53/134.7`, implicit-vs-relaxed mismatch `3.86/299.0`, and energy-force closure
mismatch `0.995/0.999`. The low-frequency projected density gradient is `1.06e-8`, marginally
above the preregistered `1e-8` parent gate. It is therefore conservatively parent-excluded, while
the evidence identifies second-response/finite-difference instability rather than branch-energy
competition as the dominant pathology.

The complete formal val8 audit gives 19/32 stable directions. The all-four v2 rule admits 2/8
parents (`0027926`, `0132890`); hierarchical v3 admits 3/8 (`0000751`, `0027926`,
`0132890`). `0044504` remains excluded under both rules. This is validation-only diagnostic
evidence; complete val20 aggregation is still pending.

## Training variants

All variants start from the same random1000 actual-force-weight-1.0 epoch-9 checkpoint, use
`lr=2e-5`, batch size 4, gradient accumulation 8 and 1200 optimizer steps.

| variant | curvature term | exact definition |
|---|---|---|
| A | none | energy + density gradient + scalar learned-energy force |
| B | energy secant | paired learned `kin_plus_xc` scalar-energy secant, weight 0.01 |
| C | direct HVP | fixed-density learned-energy HVP to analytic PBE complete-total HVP target |
| D | relaxed-force secant | scalar learned-energy endpoint-force secant to converged KS/PBE complete-total endpoint-force secant |
| E | implicit teacher HVP | fixed-density learned-energy HVP to branch-mean baseline OFDFT implicit complete-total HVP target, at most 10 parents |

E is a surrogate label experiment. It is not an end-to-end implicit prediction path: geometry
integral construction, density optimization and SciPy MINRES still detach their graphs. D is also
not a density-relaxed OFDFT prediction, although its target is a complete-total relaxed KS/PBE
force secant. These names must remain explicit in every table.

The four-batch replay cycle is pair/HVP/ordinary/ordinary. Pair and HVP batches are disjoint, so
81.25% of graphs are ordinary train800 energy/force replay. HVP directions are resampled by epoch.
Direct HVP starts at step 100, ramps for 200 steps, uses a soft normalized-loss cap of 10 and sparse
GradNorm targeting an HVP/force parameter-gradient ratio of 0.25. The screen scans curvature
weights `{1e-5,1e-4,1e-3}` and reference floors `{1e-2,5e-2}`.

## Validation funnel

1. validation energy/force and base-geometry train700 forgetting;
2. v3-stable-parent/direction fixed-density HVP against analytic PBE as a fast proxy;
3. v2/v3 stable-only strict complete-total density-relaxed HVP;
4. full complete-total Hessian and vibrational metrics on 5--10 representatives;
5. three seeds for shortlisted settings;
6. one Test100 confirmation only after all rules and candidates are frozen.

Promotion requires validation energy degradation no greater than 5%, train700 energy/force
degradation no greater than 5% in all three seeds, validation force and strict complete-total HVP
improvement in at least two of three seeds, majority parent and direction wins, and no full-Hessian
or vibrational regression. Small-reference directions and excluded parents are reported separately.

The single-seed screen evaluates all 20 A--E configurations. A deterministic ranking then freezes
one setting per C/D/E: energy-gate status, number of passed Tier-1 gates, worst force/HVP relative
change, their sum, energy change, then numeric weight/floor. A fallback is explicitly marked
non-promoted when no setting passes every Tier-1 gate. A/B and the frozen C/D/E settings are then
run at all three seeds. Only settings passing this three-seed funnel enter strict validation.

Strict validation uses only v3-eligible val20 parents and their passing directions. It uses
reference-density base initialization, base-density continuation at displaced geometries,
`h=1e-5 Bohr`, Adam `1e-3` to `1e-2`, Adam `3e-4` to `1e-5`, then LBFGS/Newton
refinement. Density and response convergence are hard gates. Its task table cannot be generated
until both the three-seed analysis and val20 stability analysis finish.

## Current execution state

Remote root: `/scratch/xzh/models/hvp_branch_stability/20260716`.

- 120 multi-direction PBE sidecars are complete, 480/480 directions finite;
- val8 branch audit: Slurm 1347, complete 96/96 with no failures;
- frozen pause boundary: old Slurm 1443/1464 stopped at 830/1200 complete;
- node01-only missing train100 array: Slurm 2330, 370 tasks with at most 8-way concurrency;
- node01-only validation remaining12: Slurm 2331, dependent on 2330;
- node01-only val/train postprocessing: Slurm 2332/2333;
- D/E two-step real-data smoke: Slurm 1717 and 1718;
- A-E screen, Tier-1 evaluation, three-seed replication and strict/full validation remain
  dependency-chained after 2332/2333 and default to node01.

The frozen pause inventory has 830 validated complete tasks, 370 missing tasks and 40 partial
directories; no completed task failed validation. The first 8 resumed tasks started on node01
with zero failures. All 20 validation parents
have PBE Hessians, four-direction sidecars and base labels. No A-E model metric, shortlist or
promotion claim exists. Test100 remains unread.

The machine-readable archive is
`/scratch/xzh/models/hvp_curvature_v1/20260716/pause_node01_20260716` and is documented in
`docs/qm9_hvp_curvature_pause_node01_20260716.md`. Future computation defaults exclusively to
node01 with at most eight concurrent GPU tasks unless the user explicitly authorizes another node.
The node01-only resume script has been run once for this resume event.

Using completed-task medians matched by direction, initialization and atom count, the frozen 370
missing tasks represent about 21.7 GPU-hours: 2.71 ideal node01 wall-hours or about 3.4 hours with
a 25% long-tail allowance. This estimate covers train100 completion only, not val12 or A--E.

Two real-data GPU smoke issues were found and fixed before the formal screen. Hydra optional
dataset keys now use `++` add-or-override syntax. Checkpointing now updates `last.ckpt` every 200
optimizer steps, so a run ending at step 1200 cannot silently evaluate an older epoch checkpoint.
A checkpoint/logging smoke at step 2 produced a 224,572,200-byte `last.ckpt`, final throughput
step 2, four per-loss gradient-norm tags and six pairwise gradient-cosine tags. C and E active-HVP
smokes both completed second-order backward; representative nonzero raw HVP losses were 0.157 and
1.60. D completed with nonzero relaxed-force-secant loss. These are execution checks, not model
quality results.

A read-only interim aggregation over the first 38 parents with all 12 branch tasks complete found
66/152 stable directions, 1/38 v2 parents and 9/38 v3 parents. Direction failures were dominated by
energy-force closure (47) and step stability (41); parent-level exclusions were dominated by
density gradient (13). This supports hierarchical masking but is not the final train100 manifest
and is not used for selection.

A later read-only v3 capacity check over 59 fully complete parents found 20 stable parents and
115/236 stable directions with zero load errors. Direction exclusions were led by energy-force
closure (64), step stability (61), density gradient (39), implicit disagreement (11), branch
energy (4), and branch force (1). This exceeds the five-parent training capacity gate but remains
interim: no sidecar or hyperparameter was frozen from this partial result.

The two 600-task Slurm arrays were initially capped at 20 tasks each. Once their remaining queues
became imbalanced, both throttles were raised to 40 while the five-node GPU pool continued to cap
total concurrency at 40. The scheduler immediately rebalanced active tasks from fixed `20/20` to
`23/17`; this changes only scheduling and avoids idle GPUs during array tails.

The expensive strict stages now isolate failures at one run x one direction x one molecule.
Strict-HVP and full-Hessian analyses execute under Slurm `afterany`, preserve all successful raw
rows, and write `strict_failures.csv` or `full_hessian_failures.csv`. A run is excluded from every
mean and promotion gate unless every preregistered task is present, finite, and matched to the
same-seed A baseline; incomplete data can never authorize Test100. Multiseed checkpoint reuse now
requires both `last.ckpt` and a throughput final step exactly equal to the requested 1200 steps.
The resume orchestrator explicitly propagates the newly submitted validation postprocessing job
ID, so stable-only strict validation cannot start before the frozen val20 manifest exists and the
cancelled job 1797 cannot be reused accidentally. Twenty focused remote
tests covering stability, screen/multiseed matching, strict/full failure handling, complete-total
HVP, and vibration artifact loading pass.

The train100 stability postprocessor no longer retains lazy `NpzFile` handles for every task.
It indexes only paths, loads the three branches for one parent-direction under context managers,
and closes them before advancing. A real read-only check over 682 completed formal records held
the process file-descriptor count at `4 -> 4 -> 4` through inventory and one full direction gate;
this removes the otherwise likely 1200-file `EMFILE` failure without keeping all Coulomb matrices
resident in memory.
The dependent implicit-teacher builder was updated to read only `implicit_hvp` from each branch
path. A real val8 smoke generated one finite `(4, 9, 3)` teacher sidecar for `0000751` with its
3/4 stability mask. The formal train100 stable/implicit output directories were absent at the
check, so the final postprocessor starts without stale sidecars.
Filtered stable sidecars now carry an authoritative `manifest.json` with parent IDs, direction
masks and per-file SHA256. A full val8 smoke produced exactly three v3 sidecars and three manifest
entries. Train100 postprocessing requires the NPZ count to equal the manifest count before any
screen job is submitted.
Implicit teacher manifests now also include each output SHA256 and mask plus hashes of the source
stability summary and parent/direction CSVs. A real `0000751` smoke matched its file hash and
`[1,1,1,0]` mask. Both stable and implicit manifest counts are hard preconditions for training.

### Execution update: 2026-07-17 10:22 Asia/Singapore

The node01-only resume completed train100 at 1200/1200 task summaries and completed all 144
remaining val20 branch tasks. The 20-run single-seed screen and 15-run three-seed comparison also
completed. The three-seed Tier-1 summary is
`/scratch/xzh/models/hvp_curvature_v1/20260716/multiseed_analysis/summary.json`.

Only direct-HVP C at weight `1e-4` and reference floor `0.01` passed the complete multiseed Tier-1
gate. Relative to matched-seed A, its mean validation energy, force, and fixed-density HVP MAE
changes were `-15.11%`, `-0.70%`, and `-0.86%`; direction and parent win fractions were 54.17% and
66.67%. These remain fixed-density validation metrics. Test100 is still unread.

Strict complete-total validation selected the three C seeds plus the three matched A seeds over
seven stable validation parents and 24 stable parent-directions, producing 144 atomic tasks in
`strict_tasks_v1.tsv`. Slurm array 2967 completed 144/144 on node01 with zero failed tasks; analysis
job 2968 ran under `afterany`, so failures could not be hidden.

The first multiseed Tier-1 evaluation array failed before loading checkpoints because its TSV had
the additional `screen_selection_status` column. The shell parser consumed the remainder of the
row into `run_name`. `slurm_qm9_hvp_curvature_stage1_eval.sbatch` now reads the optional eighth
field; replacement jobs 2950/2951 completed 15/15 evaluations and analysis without modifying any
checkpoint, label, or prior evaluation artifact.

#### Current Tier-1 assessment

Three-seed validation means for the frozen setting of each variant are:

| variant | val energy MAE | val force MAE | fixed HVP MAE | fixed HVP rel. Fro | Tier-1 result |
|---|---:|---:|---:|---:|---|
| A | 0.0369873 | 0.00253073 | 0.0277952 | 1.04605 | matched baseline |
| B | 0.0292708 | 0.00254905 | 0.0276814 | 1.06748 | reject |
| C, `1e-4` | 0.0289434 | 0.00251277 | 0.0275547 | 1.04609 | strict candidate |
| D, `1e-5` | 0.0295111 | 0.00251438 | 0.0276524 | 1.05634 | reject |
| E, `1e-3` | 0.0276580 | 0.00252113 | 0.0275940 | 1.09099 | reject |

C improves force and fixed-HVP MAE in two of three seeds and wins 13/24 eligible directions in
every seed. Its parent wins are 4/7, 5/7 and 5/7. Energy MAE improves in all three seeds, but the
reported 15.1% mean relative improvement is enlarged by an unusually weak seed-314159 A baseline
(`0.05772` versus `0.03535`); the median absolute energy MAEs are `0.02713` for A and `0.02588`
for C. The curvature effect is therefore directionally repeatable but quantitatively small:
fixed-HVP MAE improves by 0.86% on average while aggregate relative Frobenius is unchanged.

Direct HVP adds about 14% training wall time and reduces throughput by about 11.6% versus A.
Mean peak GPU memory rises from about 0.93 GiB to 1.37 GiB. The recorded C HVP gradient norm is
only about 2.4% of the force gradient norm even though the GradNorm multiplier is pinned at its
cap of 5. Average HVP-versus-energy/force gradient cosines are near zero, but individual force-HVP
cosines span roughly `-0.60` to `+0.55`. This indicates weak average conflict with substantial
batch variability, and also explains why the present HVP effect is modest.

Strict complete-total HVP passed its preregistered gate. All six runs were numerically complete.
C improved strict HVP MAE in all three seeds and had majority direction and parent wins in two of
three seeds. Mean strict HVP MAE changed from `0.128389` for A to `0.126903` for C, a 1.16%
improvement; mean strict relative Frobenius changed from `31.5042` to `31.2074`. The large relative
errors reflect small-reference directional responses and reinforce that this is still a weak
absolute-curvature model despite the repeatable relative gain.

Analysis 2968 froze the median strict-HVP seed-314159 C run and its matched A run. Full-Hessian
array 3112 contains five molecules (`natoms=9,12,17,18,20`) for each run, ten tasks total. At
2026-07-17 12:14 Asia/Singapore, all ten tasks and analysis 3121 completed with zero failures and
all displaced densities below the strict `1e-8` projected-gradient threshold.

#### Final validation-only full-Hessian assessment

| metric | matched A | C `1e-4` | formal mean paired change |
|---|---:|---:|---:|
| full-Hessian MAE | 0.0754730 | 0.0728360 | -2.73% |
| full-Hessian RMSE | 0.217591 | 0.211223 | -2.83% |
| relative Frobenius | 2.76880 | 2.69461 | -2.83% |
| antisymmetric/symmetric Frobenius | 0.002660 | 0.003326 | +17.22% |
| frequency MAE, cm-1 | 2852.20 | 2790.77 | -2.14% |
| frequency RMSE, cm-1 | 3652.95 | 3597.56 | -1.52% |
| mean mode overlap | 0.63291 | 0.63705 | +0.00414 |
| model/PBE imaginary-count error | 173 | 172 | -1 mode |

C wins Hessian MAE on 4/5 molecules. `0000751`, `0132890`, `0039447` and `0023317` improve by
7.26%, 2.22%, 3.16% and 2.50%; the preregistered candidate-regression case `0027926` worsens by
1.47%. Frequency MAE improves on all five molecules. However, the pairwise
antisymmetric/symmetric Frobenius ratio worsens on four molecules and by 17.22% on average,
exceeding the frozen 5% non-regression limit. This is the only formal full-Hessian gate failure,
but it is physically important and cannot be replaced by post-hoc symmetrization.

Absolute physical quality remains poor despite the relative gains. Mean relative Frobenius is
about 2.7, mean frequency MAE is about 2800 cm-1, and C predicts 178 imaginary modes versus six in
PBE across five molecules. The force-versus-energy diagonal second-derivative MAE is also large
(`20.03` for C versus `21.45` for A), so the `h=1e-5 Bohr` scalar-energy second-difference audit
requires a dedicated step-sensitivity interpretation before it can be treated as a closure
measurement.

The frozen decision in `full_hessian/analysis/promotion_summary.json` is therefore:

- `full_hessian_vibration_gate: false`;
- `validation_promoted_candidates: []`;
- `test100_allowed: false`;
- `test100_evaluations_used: 0`.

Direct HVP `1e-4` is a useful diagnostic candidate and demonstrates repeatable 1--3% relative
curvature gains, but it does not satisfy the goal of improving complete-total Hessians without a
physical regression. Do not expand HVP labels or access Test100 from this checkpoint.

## Reproduction

```bash
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

python scripts/prepare_qm9_hvp_branch_stability_tasks.py \
  --protocol configs/audit/qm9_hvp_branch_stability_v2.yaml \
  --subset file \
  --molecules /scratch/xzh/models/hvp100/20260716/selection/train_parent_ids.txt \
  --output /scratch/xzh/models/hvp_branch_stability/20260716/formal_train100_v2/train100_tasks.tsv

python scripts/prepare_qm9_hvp_curvature_screen.py \
  --protocol configs/audit/qm9_hvp_curvature_training_v2.yaml \
  --output /scratch/xzh/models/hvp_curvature_v1/20260716/screen_tasks_v2.tsv
```

The task table was generated with v2 because the branch task geometry/branches are unchanged from
v2 to v3; only postprocessing aggregation changes. Raw task summaries, arrays, point curves and
resource files are preserved below the remote roots and are never rewritten into source labels.
