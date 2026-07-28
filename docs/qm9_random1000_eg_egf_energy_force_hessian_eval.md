# QM9 Random1000 EG/EGF Energy Force Hessian Evaluation

Date: 2026-07-13

Last extended: 2026-07-14

## Scope

Evaluate the two trained `QM9PBEForceRandom1000` models:

- EG baseline: `/scratch/xzh/models/train/runs/qm9_random1000_eg_e10_20260713_162829/checkpoints/epoch_009.ckpt`
- EGF lambda=1.0: `/scratch/xzh/models/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829/checkpoints/epoch_009.ckpt`

Energy and force were evaluated on the full random1000 test split available to the current dataloader. Hessian was evaluated on a 10-molecule test subset with newly generated PBE analytic Hessian references.

Force and Hessian are derived from scalar model energy autograd. No force head is used.

## Artifacts

Remote output directory:

```text
/scratch/xzh/models/eval/qm9_random1000_eg_egf_metrics/20260713_190645
```

Main outputs:

```text
summary.json
energy_force_eg.json
energy_force_egf_lam1.json
pbe_hessian_manifest_10mol.json
hessian_fixed_density_10mol.json
hessian_fixed_density_10mol.csv
```

Slurm job:

```text
job id: 480
state: COMPLETED
exit code: 0:0
node: node05
elapsed: 00:59:41
start: 2026-07-13T19:06:44
end: 2026-07-13T20:06:25
```

Scripts:

```text
scripts/launch_qm9_random1000_eval.sh
scripts/slurm_qm9_random1000_eval.sbatch
scripts/qm9_force_eval.py
scripts/qm9_pbe_hessian_reference_set.py
scripts/qm9_hessian_eval_reference_set.py
```

## Energy And Force

Energy means the configured training energy label, `e_kin_plus_xc`. Force is compared with stored PBE force labels. The current dataloader evaluates 4973 test samples because `keep_initial_guess=false` removes one initial-guess state per test label file.

| model | energy MAE | energy RMSE | energy max abs | force component MAE | force component RMSE | force vector MAE | force max component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EG | 0.027565 | 0.038543 | 0.200287 | 0.070219 | 0.094251 | 0.141909 | 0.540264 |
| EGF lambda=1.0 | 0.027027 | 0.037877 | 0.233948 | 0.003678 | 0.006132 | 0.007533 | 0.085982 |

Relative to EG:

- EGF energy MAE is slightly lower: about `1.95%`.
- EGF energy RMSE is slightly lower: about `1.73%`.
- EGF energy max absolute error is worse: `0.233948` vs `0.200287`.
- EGF force component MAE is about `19.1x` lower.
- EGF force component RMSE is about `15.4x` lower.
- EGF force vector MAE is about `18.8x` lower.

Force eval coverage:

```text
samples_evaluated: 4973
atoms_evaluated: 87075
failures: 0
```

The EG worst force samples are concentrated around `0044411`. The EGF worst force samples are much smaller and include `0045927`, `0129309`, and `0000777`.

## Hessian Definition

Reference:

- PBE analytic Hessian from existing `.chk`, generated separately under `pbe_hessians/`.
- 10 smallest available `sample_id=0` molecules from the random1000 test split.

Model Hessian:

- fixed-density finite difference of derived force;
- force is `F_pred = -dE_pred/dR`;
- finite-difference displacement: `1e-3`;
- `scf_iteration=1`;
- no density optimization after displacement.

This is a fixed-density Hessian proxy, not a density-relaxed physical OFDFT Hessian.

## Hessian Summary

| model | n success | n failed | Hessian MAE | Hessian RMSE | relative Frobenius | symmetry max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EG | 10 | 0 | 0.075317 | 0.240548 | 2.256694 | 0.000521 |
| EGF lambda=1.0 | 10 | 0 | 0.019962 | 0.059027 | 0.552192 | 0.000190 |

Relative to EG:

- EGF Hessian MAE is about `3.77x` lower.
- EGF Hessian RMSE is about `4.08x` lower.
- EGF relative Frobenius error is about `4.09x` lower.
- EGF model Hessian symmetry error is also lower.

## Per-Molecule Hessian Metrics

| molecule | natoms | EG MAE | EGF MAE | EG rel Fro | EGF rel Fro |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0000777 | 7 | 0.103061 | 0.031231 | 2.381694 | 0.540738 |
| 0021889 | 11 | 0.058263 | 0.019399 | 1.703300 | 0.517970 |
| 0129309 | 11 | 0.070664 | 0.027946 | 2.374446 | 0.651542 |
| 0132081 | 11 | 0.044246 | 0.022309 | 1.637205 | 0.651688 |
| 0000242 | 12 | 0.077459 | 0.013906 | 2.463326 | 0.495255 |
| 0003027 | 12 | 0.096219 | 0.012323 | 2.276565 | 0.359525 |
| 0025417 | 13 | 0.059107 | 0.015134 | 2.387012 | 0.518644 |
| 0029684 | 13 | 0.054127 | 0.016452 | 1.621029 | 0.541258 |
| 0044411 | 13 | 0.094561 | 0.021973 | 2.968182 | 0.732954 |
| 0064551 | 13 | 0.095466 | 0.018943 | 2.754181 | 0.512343 |

EGF is better than EG on all 10 Hessian reference molecules for MAE and relative Frobenius error.

## Time Cost

| stage | wall time | max RSS |
| --- | ---: | ---: |
| EG energy/force eval | 1:25.88 | 1.48 GB |
| EGF energy/force eval | 1:28.18 | 1.49 GB |
| PBE Hessian reference, 10 mol, 8 workers | 52:57.51 | 4.68 GB |
| fixed-density Hessian eval, EG+EGF, 10 mol | 3:48.18 | 1.74 GB |

PBE Hessian reference generation dominated total time. Individual PBE Hessian reference times ranged from `655.8 s` for 7 atoms to `2296.5 s` for 13 atoms.

## Conclusion

On this random1000 evaluation:

- EGF lambda=1.0 gives a very large force improvement over EG.
- EGF lambda=1.0 also gives a clear fixed-density Hessian proxy improvement over EG on all 10 tested molecules.
- Energy does not degrade in MAE/RMSE; it is slightly better than EG on average, though EGF has a larger energy max absolute error.

The Hessian conclusion is limited to a 10-molecule fixed-density proxy benchmark. It should not be stated as a density-relaxed physical Hessian result.

## 2026-07-14 Test100 GPU4PySCF Hessian Extension

The fixed-density Hessian benchmark was extended from 10 molecules to all 100 molecules in the
random1000 molecule-grouped test split. PBE analytic Hessian references were generated or loaded
from cache with GPU4PySCF, while model Hessians retained the same fixed-density finite-difference
definition used above.

### GPU4PySCF Validation

Before the full run, GPU4PySCF was checked against five existing CPU PySCF Hessian references:

```text
/scratch/xzh/models/eval/qm9_random1000_gpu4pyscf_hessian_validation/20260714_131826
```

All five GPU Hessians were finite and passed the configured agreement thresholds:

- maximum GPU-vs-CPU relative Frobenius error: `2.179982e-4`;
- maximum GPU-vs-CPU absolute element error: `3.416659e-4`;
- thresholds: relative Frobenius `1e-3`, maximum absolute element error `2e-3`;
- GPU4PySCF elapsed time was `29.98-53.71 s` per validation molecule, mean about `39.8 s`.

For context, the prior CPU PySCF reference time for molecule `0000777` was `655.8 s`, compared
with `36.2 s` in the GPU validation. This is an indicative same-molecule comparison, not a
controlled end-to-end benchmark across all molecule sizes.

### Full Test100 Artifacts

Final Slurm run:

```text
job id: 485
node: node04
GPUs: 8 x A100
output: /scratch/xzh/models/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/20260714_135600
summary: /scratch/xzh/models/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/20260714_135600/summary.json
PBE cache: /scratch/xzh/models/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians
```

Reference generation completed for `100/100` molecules with no failures. At the beginning of the
final run, `92` references were already cached and GPU4PySCF computed the remaining `8`.

An earlier job, `484`, covered only `80/100` molecules because each of eight externally sharded
GPU processes inherited the reference script's default `--max-molecules=10`. The launcher now
passes each shard's actual molecule count to `--max-molecules`. Job `484` is retained only as a
debug artifact and must not be used as the final Test100 result.

### Test100 Hessian Results

| model | n success | n failed | Hessian MAE | Hessian RMSE | relative Frobenius | symmetry max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EG | 100 | 0 | 0.066706 | 0.200525 | 2.921985 | 0.000408 |
| EGF lambda=1.0 | 100 | 0 | 0.010798 | 0.034852 | 0.479906 | 0.000129 |

Relative to EG, EGF lambda=1.0 has approximately `6.18x` lower Hessian MAE, `5.75x` lower
Hessian RMSE, and `6.09x` lower relative Frobenius error. The EG/EGF ordering therefore remains
stable when the benchmark is expanded from 10 to all 100 test molecules.

### Test100 Time Cost

| stage | parallelism | wall time | result |
| --- | ---: | ---: | --- |
| PBE reference stage | 8 GPU shards | 168 s | 100/100 success; 92 cached, 8 computed |
| model fixed-density Hessian stage | 8 GPU shards | 305 s | EG and EGF, 100/100 each |
| measured stage total | 8 x A100 | 473 s (about 7.9 min) | excludes small launch/merge overhead |

The PBE stage time in the final run is cache-assisted and must not be interpreted as the cost of
computing 100 new PBE Hessians. The five-molecule validation provides the cleanest current
GPU4PySCF per-molecule timing sample.

### Updated Conclusion

On the full 100-molecule random1000 test split, EGF lambda=1.0 retains a large and consistent
advantage over EG for the fixed-density Hessian proxy, in addition to its force improvement.
This materially strengthens the random1000 early-pilot result, but it remains a fixed-density,
finite-difference proxy with `scf_iteration=1`. It is not a density-relaxed physical OFDFT
Hessian result and must not be generalized to final full-QM9 or P1/P2 conclusions.

## 2026-07-14 Test100 Density-Relaxed Extension

The same 100 test molecules were subsequently evaluated with strict density optimization at every
finite-difference displacement. Full details are in
`docs/qm9_random1000_test100_density_relaxed_hessian.md`.

| model | density-relaxed MAE | RMSE | relative Frobenius | symmetry max abs |
| --- | ---: | ---: | ---: | ---: |
| EG | 0.070755 | 0.207097 | 3.046408 | 0.444027 |
| EGF lambda=1.0 | 0.012074 | 0.037043 | 0.516296 | 0.126402 |

All `21,276/21,276` displaced density optimizations and `200/200` base optimizations reached the
strict `1e-4` projected-gradient threshold. EGF is better than EG on MAE, RMSE, and relative
Frobenius for every `100/100` molecule.

The 8-A100 job took `04:12:00`, about `49.6x` the fixed-density FD model stage. Density relaxation
slightly worsened average agreement with PBE for both models and substantially increased Hessian
antisymmetry. This supports stable EGF ordering after density optimization, but not a claim of a
high-accuracy analytic total-OFDFT physical Hessian.
