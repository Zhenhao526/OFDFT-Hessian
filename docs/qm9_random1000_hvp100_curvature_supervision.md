# QM9 random1000 HVP100 Curvature-Supervision Experiment

Last updated: 2026-07-16

## Scope and decision contract

This experiment tests whether a small amount of direct directional Hessian supervision can
simultaneously improve validation force error and strict density-relaxed complete-total OFDFT
curvature. It is a random1000 pilot, not a full-QM9 or final P1/P2 result.

The frozen promotion contract is:

1. validation energy MAE may increase by at most 5% relative to matched-seed A;
2. validation force MAE and strict complete-total HVP MAE must improve in at least two of three
   seeds;
3. most validation molecules must improve, including explicit reporting of small-reference cases;
4. a promoted model must not regress a three-molecule full Hessian, frequency, imaginary-mode and
   mode-overlap audit;
5. Test100 remains unread unless all preceding rules pass and a candidate is frozen.

No Test100 label, prediction or metric was used for parent selection, training, hyperparameter
selection, stage-1 screening, strict validation or the full-Hessian diagnostic. Test parent IDs
were read only for a zero-overlap assertion.

## Frozen data

The source is `QM9PBEForceRandom1000PairedAug`. The original random1000 parent split contains
800/100/100 train/validation/test parents. Every original parent has four labelled geometries; the
augmented train source adds one exact `R-/R+` pair per train parent.

The HVP experiment freezes 100 stratified parents from the original train800. Stratification uses
atom-count quartile, coarse elemental composition and a combined difficulty quartile formed from
baseline force error and an existing paired force-secant curvature error. All six geometries of a
selected train parent remain in train. Validation retains 100 independent parents and Test100 is
frozen.

| artifact | count or SHA256 |
|---|---:|
| selected train parents | 100 |
| selected train geometry paths | 600 |
| validation parents/geometries | 100 / 400 |
| train/validation parent overlap | 0 |
| train/Test100 parent overlap | 0 |
| validation/Test100 parent overlap | 0 |
| selected-parent manifest SHA256 | `8982fc6349b520b9b98f52fa684e81558b6afab16d14a100f3c3843608503f0b` |
| selected train ID SHA256 | `b1707a218fa4b01b16e22b9594d3bd871654030e62e13533bcf5a37eed5914d7` |
| experiment split pickle SHA256 | `ce522d7ddc312c303a0066e4759b15a18d8fc36a8d469398fa0629ab30f412d8` |
| experiment split YAML SHA256 | `f0896877a6607ec16caf27a6457538fdb29d4727c6866a56fc397eb03066e9b2` |
| train800 difficulty CSV SHA256 | `82d458e184e69c9c247aad8c46b657687b8826f019dfd0a593d2a5308abacad3` |
| validation20 representative manifest SHA256 | `10d00651eedd0dfec853b70edaffc21a2162bd427667ed0e7fc62157e0e0979d` |

Persistent selection root:

```text
/scratch/xzh/models/hvp100/20260716/selection
```

## HVP reference definition

PBE analytic Cartesian Hessians were generated for all train100 base geometries and the frozen
validation20 representatives with GPU4PySCF. All 120 succeeded and are finite. The combined
manifest SHA256 is
`bee31003d563e8cd2fe516a6e90b2255d228b8ad6446a6b026dfd8005bfdc672`.
The 120 analytic PBE Hessians required `2.21 GPU-hours` in summed evaluator time: mean 66.4 s,
median 67.1 s and maximum 102.6 s per molecule. Cached records are excluded from this timing.

Each base structure has one deterministic unit direction. The 120 directions contain 42 existing
paired perturbations, 44 projected random internal directions, 23 bond stretches and 11 low-mode
directions. Mass-weighted translation and rotation components are removed. The sidecar manifest
SHA256 is `4ac88f11740a9f35e9cdc0cbdbb9e27cb6064db08864dff3d77f36249469b1cd`.

The training target is

```text
HVP_target = H_PBE,total * v
HVP_pred   = d/dR [(d E_learned / dR) . v]
```

This is deliberately called a `learned-energy fixed-density HVP to PBE-total target` surrogate.
It is not a density-relaxed complete-total OFDFT HVP. The force baseline already compares the
learned scalar component derivative with a PBE total force, and this experiment tests the direct
second-order analogue of that objective.

An attempted baseline-anchored fixed-total correction was rejected before formal training. On
`0000023`, the PBE HVP RMS was `0.60897`, the learned trainer-path HVP RMS was `0.68459`, but the
putative correction RMS was `2424.37`. The projected density gradient near 85 proves that the PBE
training density is not stationary for the learned model. Applying that correction would create a
roughly 4000-fold label-scale error. Formal sidecars therefore use correction mode `none`.

The authoritative physical validation is separate:

```text
strict complete-total HVP = finite difference of complete scalar-derived total OFDFT force
                            after strict density relaxation at R+/-h
```

It includes learned, Hartree, electron-nuclear, nuclear-nuclear, overlap/Pulay, constraint and
density-response contributions through the audited total-energy implementation.

## Loss and batch protocol

The direct HVP loss is an atom/component mean L1 error. Per graph it is divided by direction norm,
PBE reference HVP RMS with a `1e-2` floor, and atom count. Unlabelled graphs receive a false mask
and shape-compatible zero tensors.

Direct HVP autograd is eligible on 25% of batches, with an epoch-shifted deterministic phase. It is
actually executed only when an eligible batch contains a labelled graph. The formal three-seed
mean was 11.12% active batches and 3.15% active graphs. An 8-step real-GPU smoke measured 12.5%
active batches, finite third-order backward, no NaN and a saved checkpoint.

All formal runs use:

| setting | value |
|---|---:|
| source initialization | force-weight-1.0 epoch 9 checkpoint |
| optimizer steps | 600 |
| per-GPU batch | 4 |
| gradient accumulation | 8 |
| effective batch | 32 |
| seeds | 676368232, 20260716, 314159 |
| validation interval | 5 epochs |
| direct HVP weights | `1e-5`, `1e-4`, `1e-3` |
| secant weight | 0 or `0.01` |

The four ablations are:

| variant | objective |
|---|---|
| A | energy + density gradient + force |
| B | A + scalar learned-energy secant |
| C | A + direct HVP |
| D | A + secant + direct HVP |

Every run uses the same selected data, starting checkpoint, optimizer-step count, batch size,
learning-rate schedule and matched random seed. This is equal update/data exposure, not equal
FLOPs, because C/D perform third-order parameter backpropagation on active HVP batches.

Formal configuration provenance:

| file | SHA256 |
|---|---|
| `configs/ml/experiment/str25/qm9_pbe_force_full_egf_hvp.yaml` | `c4b1a73273707a46d6b2532bf54cbfe789e3e6204e614982e965c2fd18d8e876` |
| `configs/ml/experiment/str25/qm9_pbe_force_full_egf_energy_secant_hvp.yaml` | `95a3f12de12c3a3418d3d649e7727b7590d377e7f2a7b66beec29fa5e69b189e` |
| `configs/ml/model/loss_function/l1_force_hvp.yaml` | `e895ece91cd1f9ccf8dba185ac0f9b17a766f5c25bcaefa9d8a7b5c008e31db7` |
| `configs/ml/model/loss_function/l1_force_energy_secant_hvp.yaml` | `e348a482a166f6cdfcf01c7269bd4bb64dcc04092431c4c5a4393fe3b8e51540` |
| `scripts/launch_qm9_hvp100_train.sh` | `a5ed030707b1abb72c3dd912acb7c8b6089dff50981dc14c789315d1b3a49c33` |
| `scripts/slurm_qm9_hvp100_ablation_gated25_array.sbatch` | `4622c5085fee09900379e82fcad2104fb721ca55b866bdbf996505e4a9f31182` |

Core implementation is in `OFData` sidecar loading, `DirectionalHVPLoss`, the scalar-energy HVP
path in `MLDFTModule`, and the throughput callback's HVP timing field. The HVP path computes
`-d(F.v)/dR` from the same scalar model energy and does not introduce a force head. Seven focused
loss/autograd tests pass locally and remotely; the real eight-step GPU smoke completed forward,
third-order backward, logging and checkpoint save with finite values.

## Stage-1 validation

Energy is the learned `kin_plus_xc` target MAE in Hartree. Force is PBE total-force component MAE
in Hartree/Bohr. Fixed HVP is the fast surrogate component MAE in Hartree/Bohr squared. Values are
three-seed means; changes are relative to matched-seed A.

| variant | HVP weight | energy MAE | energy change | force MAE | force change | fixed HVP MAE | HVP change | molecule win fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 0.030515 | 0.0% | 0.002660 | 0.0% | 0.023816 | 0.0% | reference |
| B | 0 | 0.055597 | +83.04% | 0.002663 | +0.13% | 0.023682 | -0.57% | 0.60 |
| C | `1e-5` | 0.035776 | +18.18% | 0.002699 | +1.46% | 0.024157 | +1.44% | 0.45 |
| C | `1e-4` | 0.036932 | +20.82% | 0.002672 | +0.44% | 0.023924 | +0.48% | 0.53 |
| C | `1e-3` | 0.034398 | +13.17% | 0.002796 | +5.12% | 0.023553 | -1.12% | 0.63 |
| D | `1e-5` | 0.032518 | +6.80% | 0.002689 | +1.07% | 0.023732 | -0.36% | 0.53 |
| D | `1e-4` | 0.054320 | +76.58% | 0.002724 | +2.40% | 0.023798 | -0.07% | 0.57 |
| D | `1e-3` | 0.044855 | +47.29% | 0.002690 | +1.11% | 0.023952 | +0.56% | 0.47 |

No direct-HVP configuration passes stage 1. C `1e-3` is the best C fixed-HVP diagnostic despite
force and energy regression. D `1e-5` is the best D diagnostic because it has the least severe
joint tradeoff. Neither is a promotion candidate.

## Gradient scale and conflict

Sparse weighted parameter-gradient norms were recorded on three seeds at four training snapshots.
Typical HVP-to-force norm ratios are about 0.07%, 0.6% and 5.7% at weights `1e-5`, `1e-4` and
`1e-3`, respectively. Thus the two smaller weights are genuinely weak; `1e-3` is large enough to
affect optimization but still does not improve held-out force or fixed HVP.

A separate four-batch train100 audit computes pairwise gradient cosines without Test100:

| run | pair | mean cosine | negative-batch fraction |
|---|---|---:|---:|
| C `1e-3` | HVP vs force | +0.284 | 0.25 |
| C `1e-3` | HVP vs density gradient | -0.076 | 0.75 |
| C `1e-3` | HVP vs energy | +0.012 | 0.00 |
| D `1e-5` | HVP vs force | +0.117 | 0.25 |
| D `1e-5` | HVP vs density gradient | +0.112 | 0.00 |
| D `1e-5` | HVP vs secant | +0.012 | 0.50 |

The direct HVP and force gradients are not systematically opposed. C has mild conflict with the
density-gradient objective, but label semantics, sparse directional coverage and generalization
are stronger concerns than a simple HVP-vs-force sign conflict.

## Training cost

All costs below are three-seed means on one A100. Direct HVP adds 6% to 10% wall time and about 46%
peak allocated GPU memory at the measured 11.12% active-batch rate.

| variant | HVP weight | wall (s) | samples/s | peak GPU MiB | wall/A | memory/A |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 948 | 21.56 | 832 | 1.000 | 1.000 |
| B | 0 | 905 | 22.61 | 832 | 0.954 | 1.000 |
| C | `1e-5` | 1010 | 20.26 | 1219 | 1.065 | 1.464 |
| C | `1e-4` | 1001 | 20.45 | 1219 | 1.057 | 1.464 |
| C | `1e-3` | 1034 | 19.69 | 1219 | 1.090 | 1.464 |
| D | `1e-5` | 1007 | 20.36 | 1219 | 1.062 | 1.464 |
| D | `1e-4` | 1034 | 19.75 | 1219 | 1.090 | 1.464 |
| D | `1e-3` | 1044 | 19.54 | 1219 | 1.101 | 1.464 |

The measured HVP-autograd timing field is about 5.2 ms for the interval's last batch; full-run wall
and throughput are the authoritative cost measures because interval timing does not average every
active/inactive batch event.

The 24 formal one-GPU training runs consumed `6.65 A100 GPU-hours` in aggregate. This excludes PBE
label generation and post-training physical validation.

## Strict complete-total HVP and full Hessian

The strict validation uses 20 preselected validation parents, one frozen direction per parent,
`h=1e-5 Bohr`, strict density relaxation and the complete scalar-derived total OFDFT force at
both displaced geometries. All 240 model/molecule evaluations completed without a failed response
solve. The table reports three-seed means relative to matched-seed A:

| variant | HVP weight | strict HVP MAE | change | relative Frobenius | molecule win fraction | improved seeds | force-improved seeds | energy-gate seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 11.8529 | 0.0% | 650.75 | reference | 0/3 | 0/3 | 3/3 |
| B | 0 | 11.8510 | +3.04% | 656.62 | 0.383 | 2/3 | 0/3 | 1/3 |
| C | `1e-3` | 10.2042 | **-14.51%** | 574.94 | **0.550** | **3/3** | 1/3 | 1/3 |
| D | `1e-5` | 12.6185 | +8.08% | 680.46 | 0.217 | 1/3 | 2/3 | 1/3 |

C `1e-3` therefore has a reproducible complete-total directional-curvature effect: every seed
improves strict HVP MAE. It still fails the frozen promotion rule because validation force improves
in only one seed and worsens by 5.12% on average, while energy worsens by 13.17% and passes the 5%
gate in only one seed. D `1e-5` does not reproduce the curvature gain.

The 12 strict validation20 runs consumed `10.55 A100 GPU-hours` in aggregate, averaging 52.8 min
per model/seed. Peak host RSS was 4.57 GiB and peak allocated GPU memory was 805 MiB. This strict
stage is more expensive than all formal training because every directional endpoint requires a
strict response-aware density solve.

The strict mean is dominated by the hard case `0044504`, whose PBE reference HVP RMS is only
`0.048805`. Across A seeds its component MAE is `194.1`--`275.8`, versus `156.4`--`251.4` for C.
For the diagnostic seed `676368232`, C changes this molecule from `233.13` to `196.38`, but its
relative Frobenius remains about `1.00e4`. Non-hard validation molecules have typical component
MAEs around `0.08`--`0.25`. Consequently neither mean relative Frobenius nor mean MAE alone is a
sufficient model-selection statistic; the report preserves per-molecule rows, small-reference
MAE and molecule win fractions.

The explicitly separated small-reference case is `0059830` with reference RMS `7.7e-5`. Its
three-seed mean absolute HVP error is `0.02954` for A, `0.03186` for B, `0.03316` for C and
`0.03253 Ha/Bohr^2` for D. Direct HVP supervision does not improve this case, so the lower C
aggregate is not evidence of uniform behavior at small curvature scale.

The three-molecule complete-total Hessian/frequency audit uses the matched seed `676368232` for
A, B and C `1e-3`. It is diagnostic only because no direct-HVP model passed the preceding gates.
The selected molecules are small `0000751`, median `0103559`, and hard `0044504`. Test100 remains
unread.

All `846/846` displaced points are strict at a maximum projected density-gradient norm below
`1.0e-8`. Mean full-matrix metrics are nevertheless dominated by `0044504`:

| variant | Hessian MAE | RMSE | relative Frobenius | symmetrized MAE | mean asym/sym Fro |
|---|---:|---:|---:|---:|---:|
| A | 26.1366 | 163.729 | 2590.11 | 25.9636 | 0.0981 |
| B | 26.8369 | 171.637 | 2715.23 | 26.6595 | 0.1071 |
| C `1e-3` | **23.1422** | **143.721** | **2273.55** | **22.9790** | 0.0985 |

C lowers the three-molecule mean MAE by 11.5% and RMSE/relative Frobenius by 12.2% relative to A,
but these aggregate changes must not be interpreted without the per-molecule rows:

| molecule | role | A MAE | B MAE | C MAE | C vs A | A/C relative Fro | A/C asym/sym Fro |
|---|---|---:|---:|---:|---:|---:|---:|
| `0000751` | small | 0.12537 | 0.12540 | **0.11986** | -4.40% | 2.858 / 2.740 | 0.00260 / 0.00274 |
| `0103559` | median | 0.05925 | 0.05914 | **0.05816** | -1.84% | 2.837 / 2.777 | 0.00250 / 0.00335 |
| `0044504` | hard | 78.225 | 80.326 | **69.248** | -11.48% | 7764.6 / 6815.1 | 0.289 / 0.290 |

The two ordinary molecules are smooth enough for this diagnostic and C has a small consistent
MAE improvement. The hard molecule is not a trustworthy vibrational matrix: despite strict
density residuals, maximum non-symmetry is `3655.5` for A and `2817.4 Ha/Bohr^2` for C. This is a
density-branch/second-order pathology, not residual optimization error, and postprocessing
symmetrization is not a physical repair.

Mass-weighted, translation/rotation-projected vibrational diagnostics use the symmetrized matrices
and maximum-overlap mode matching:

| variant | frequency MAE (cm^-1) | frequency RMSE (cm^-1) | mean mode overlap | model/PBE imaginary modes |
|---|---:|---:|---:|---:|
| A | 6071.5 | 15489.4 | 0.606 | 109 / 7 |
| B | 6062.9 | 15669.3 | 0.618 | 108 / 7 |
| C `1e-3` | **5957.9** | **14702.1** | **0.632** | 106 / 7 |

Excluding `0044504`, C's frequency MAE is `2847.5` versus `2871.6 cm^-1` for A, while frequency
RMSE is effectively unchanged (`3761.3` versus `3764.3 cm^-1`). The enormous imaginary-mode
excess remains. Thus the full-Hessian/frequency diagnostic is directionally consistent with the
strict HVP gain, but does not establish useful vibrational accuracy or rescue the failed
energy/force promotion gates.

The nine full matrices consumed `3.88 A100 GPU-hours`; parallel array makespan was 51:50. Mean
per-task times were 30.4 min for A, 23.0 min for B and 24.1 min for C. Peak host RSS was 3.61 GiB
and peak allocated GPU memory was 515 MiB.

## Reproduction

Remote environment:

```bash
ssh -J root@101.200.216.125 -p 2200 shenwei01@localhost
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
```

Major entry points:

```bash
# Parent difficulty, stratified selection and PBE references
python scripts/qm9_hvp_parent_difficulty.py ...
python scripts/prepare_qm9_hvp100_experiment.py ...
bash scripts/launch_qm9_hvp100_pbe_hessians.sh
bash scripts/launch_qm9_hvp100_sidecars.sh

# Formal gated 24-run array and post-training validation
sbatch scripts/slurm_qm9_hvp100_ablation_gated25_array.sbatch  # job 1021
sbatch --dependency=afterok:1021 \
  scripts/slurm_qm9_hvp100_posttrain_gated25_array.sbatch

# Validation-only analysis and strict sharded physical HVP
python scripts/qm9_hvp100_stage1_analysis.py ...
python scripts/qm9_hvp100_gradient_scale_analysis.py ...
python scripts/qm9_hvp100_training_cost_analysis.py ...
python scripts/qm9_hvp100_prepare_strict_tasks.py ...
sbatch scripts/slurm_qm9_hvp100_strict_sharded_array.sbatch  # job 1085
python scripts/qm9_hvp100_merge_strict_shards.py ...
python scripts/qm9_hvp100_strict_analysis.py ...

# Diagnostic-only full complete-total Hessians; no Test100 access
python scripts/qm9_hvp100_prepare_full_hessian.py ...
sbatch --array=0-8%9 scripts/slurm_qm9_hvp100_full_hessian_task.sbatch  # job 1148
python scripts/qm9_hvp100_full_hessian_merge.py ...
python scripts/qm9_hessian_vibrational_metrics.py ...
```

Formal result root:

```text
/scratch/xzh/models/hvp100/20260716_gated25v2
```

Primary machine-readable and plot artifacts are:

- `stage1_runs.csv`, `stage1_groups.csv`, `stage1_per_molecule.csv` and
  `stage1_force_hvp_tradeoff.png`;
- `gradient_norm_observations.csv`, `gradient_norm_groups.csv` and
  `gradient_norm_summary.json`;
- `training_cost_runs.csv`, `training_cost_groups.csv` and `training_cost_summary.json`;
- `strict_runs.csv`, `strict_groups.csv`, `strict_per_molecule.csv`,
  `strict_force_hvp_tradeoff.png` and the log-scale `strict_per_molecule_scatter.png`;
- `full_hessian/analysis/full_hessian_per_molecule.csv`, `full_hessian_points.csv`,
  `full_hessian_resources.csv`, `full_hessian_summary.json` and `full_hessian_diagnostic.png` for
  the diagnostic full matrices;
- `full_hessian/vibrational/per_molecule_vibrational_metrics.csv` and `summary.json` for frequency,
  imaginary-mode and mode-overlap diagnostics.

Final summary SHA256 values are:

| summary | SHA256 |
|---|---|
| stage 1 | `11ea6a45e8890859f14d544d4044827a82e369c54c68090e6bebeea50a3f34de` |
| strict HVP | `518fdde1e303dec070d5e385988705845794fee13ff71cfd2c65f1e5f56ecb3c` |
| full Hessian | `69a8355a5e8bdfa771aa954da87167071aaf004c03754d044063c6709fc6ae92` |
| vibrational | `5f6641c47d1b77832704d47574976ac48a44d3df017cac9a9a00e545f7724521` |

A small local mirror, excluding checkpoints and large NPZ caches, is retained at
`_runtime/remote_artifacts/qm9_hvp100/`. It includes all 24 resolved Hydra config/override
snapshots under `20260716_gated25v2/run_configs/`.

The earlier ungated pilot and the first short gated launch are retained under distinct run names.
They are implementation diagnostics and are excluded from every formal table.

## Decision

No configuration currently satisfies the joint energy, force and strict complete-total HVP
promotion contract. C `1e-3` proves that direct HVP supervision can improve held-out physical
curvature, but it does not improve force simultaneously and exceeds the energy tolerance. The
three-molecule full-Hessian diagnostic cannot promote a model that already failed these frozen
gates. Therefore this experiment does not justify expansion to 400--800 parents or a one-shot
Test100 confirmation. No Test100 label, prediction or metric was read, and no Test100 evaluation
job was submitted for this experiment.
