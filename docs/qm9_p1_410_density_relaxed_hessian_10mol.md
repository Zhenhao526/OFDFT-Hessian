# QM9 P1-410 Density-Relaxed Hessian 10 Molecule Confirmatory Run

Date: 2026-07-07

## Scope

This report extends the P1-410 density-relaxed derived-force Hessian evaluation from 5 to 10 PBE Hessian reference molecules.

Constraints kept:

- No 1000 molecule label generation.
- No deletion of existing labels, checkpoints, or Hessian references.
- No independent force head.
- Force is derived from scalar model energy: `F_pred = -dE_pred/dR`.
- Conclusion is limited to the P1-410 early pilot.

Evaluation definition:

- For each finite-difference displaced geometry, first run OFDFT density optimization.
- Then compute model force by autograd from scalar energy.
- Then build Cartesian Hessian by finite difference of derived force.
- This remains a density-relaxed derived-force Hessian evaluation, not an analytic full total OFDFT Hessian with all classical nuclear derivatives.

## Run Configuration

Output prefix:

- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_relaxed_hessian_10mol/density_relaxed_10mol_basewarm_twostage_strict`

Command settings:

| item | value |
| --- | --- |
| evaluator | `scripts/qm9_hessian_density_relaxed_eval.py` |
| manifest | `_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json` |
| dataset | `_runtime/qm9_p1/QM9PBEForcePilot` |
| max molecules | 10 |
| displacement | `1e-3` Bohr |
| initialization | `sad_default` |
| base-density warm-start | enabled |
| stage 1 | Adam `lr=1e-3`, `max_cycle=1000`, threshold `1e-2` |
| stage 2 / fallback | Adam `lr=3e-4`, `max_cycle=10000`, threshold `1e-4` |
| fallback policy | `--fallback-always` |

Models:

| model | checkpoint |
| --- | --- |
| EG_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt` |
| EGF_lam1_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt` |

Wall time and resources:

| run | wall time | max RSS | exit |
| --- | ---: | ---: | ---: |
| 5 molecule cost-reduced reference | 17:01.99 | 1.60 GB | 0 |
| 10 molecule confirmatory | 33:30.03 | 1.63 GB | 0 |

## Convergence

Displacement optimization rows only:

| scope | points | strict `<1e-4` | medium `<1e-3` | mean cycles | median cycles | max cycles | fallback triggers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 molecule reference | 408 | 408 | 408 | 178.5 | 125 | 1114 | 408 |
| 10 molecule confirmatory | 924 | 924 | 924 | 168.7 | 126 | 1113 | 924 |

Per model:

| model | molecules | opt points | strict | mean cycles | max cycles | fallback triggers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EG_s3000 | 10 | 462 | 462/462 | 167.9 | 1113 | 462 |
| EGF_lam1_s3000 | 10 | 462 | 462/462 | 169.6 | 1111 | 462 |

Base warm-start overhead:

| model | base optimizations | base elapsed | mean base cycles | max base cycles |
| --- | ---: | ---: | ---: | ---: |
| EG_s3000 | 10 | 60.9 s | 733.4 | 1005 |
| EGF_lam1_s3000 | 10 | 107.1 s | 931.5 | 1013 |

## Hessian Metrics

Mean over the 10 molecule reference set:

| model | Hessian MAE | Hessian RMSE | relative Fro | symmetry max abs |
| --- | ---: | ---: | ---: | ---: |
| EG_s3000 | 1.192573 | 3.588587 | 21.397097 | 13.720957 |
| EGF_lam1_s3000 | 0.041654 | 0.106804 | 0.834243 | 0.414696 |

EGF remains substantially better:

- MAE ratio `EGF / EG`: 0.035
- RMSE ratio `EGF / EG`: 0.030
- relative Fro ratio `EGF / EG`: 0.039
- symmetry error ratio `EGF / EG`: 0.030

## 5 Molecule vs 10 Molecule

| scope | model | Hessian MAE | Hessian RMSE | relative Fro | symmetry max abs |
| --- | --- | ---: | ---: | ---: | ---: |
| first 5 | EG_s3000 | 1.804253 | 4.668673 | 23.496900 | 19.273569 |
| first 5 | EGF_lam1_s3000 | 0.035394 | 0.087596 | 0.561245 | 0.175163 |
| all 10 | EG_s3000 | 1.192573 | 3.588587 | 21.397097 | 13.720957 |
| all 10 | EGF_lam1_s3000 | 0.041654 | 0.106804 | 0.834243 | 0.414696 |
| added 5 only | EG_s3000 | 0.580893 | 2.508498 | 19.297281 | 8.168048 |
| added 5 only | EGF_lam1_s3000 | 0.047914 | 0.126014 | 1.107246 | 0.654246 |

The added 5 molecule subset is less severe for EG than the original first 5, mainly because `0000323` in the first 5 is a difficult EG outlier. Even on the added subset, EGF remains clearly better on MAE, RMSE, relative Frobenius error, and symmetry.

## Per-Molecule Metrics

| molecule | natoms | opt points/model | EG MAE | EGF MAE | EG RMSE | EGF RMSE | EG rel Fro | EGF rel Fro | EG sym | EGF sym |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0000010` | 6 | 36 | 1.073281 | 0.047405 | 5.559296 | 0.112141 | 32.669975 | 0.659014 | 52.881269 | 0.617570 |
| `0000323` | 6 | 36 | 5.888035 | 0.046206 | 11.607073 | 0.095034 | 50.076331 | 0.410006 | 27.857792 | 0.137468 |
| `0000109` | 7 | 42 | 1.742658 | 0.020991 | 5.076460 | 0.050927 | 26.113741 | 0.261974 | 15.053582 | 0.043283 |
| `0000171` | 7 | 42 | 0.105003 | 0.044959 | 0.347597 | 0.130269 | 2.936463 | 1.100503 | 0.127563 | 0.042797 |
| `0000062` | 8 | 48 | 0.212291 | 0.017409 | 0.752954 | 0.049602 | 5.688057 | 0.374707 | 0.449123 | 0.034611 |
| `0000116` | 8 | 48 | 0.493888 | 0.027463 | 2.319938 | 0.078846 | 14.145994 | 0.480770 | 21.410464 | 0.705194 |
| `0000144` | 8 | 48 | 0.340876 | 0.020812 | 2.646181 | 0.045507 | 19.352289 | 0.332808 | 0.603817 | 0.017264 |
| `0000019` | 9 | 54 | 0.340524 | 0.015712 | 1.302984 | 0.040799 | 12.389152 | 0.387929 | 0.466075 | 0.025316 |
| `0000052` | 9 | 54 | 0.716651 | 0.155935 | 2.031446 | 0.417376 | 19.395750 | 3.985005 | 18.076948 | 2.474951 |
| `0000350` | 9 | 54 | 1.012527 | 0.019646 | 4.241941 | 0.047542 | 31.203219 | 0.349717 | 0.282934 | 0.048503 |

Every molecule shows lower Hessian MAE and RMSE for EGF lambda=1.0 than EG. The same is true for relative Frobenius error and symmetry max abs error in this 10 molecule set.

## Per-Molecule Time

| molecule | EG elapsed | EGF elapsed |
| --- | ---: | ---: |
| `0000010` | 64.3 s | 87.2 s |
| `0000323` | 118.1 s | 145.3 s |
| `0000109` | 63.3 s | 131.2 s |
| `0000171` | 49.2 s | 98.7 s |
| `0000062` | 55.5 s | 93.5 s |
| `0000116` | 82.0 s | 120.6 s |
| `0000144` | 58.6 s | 76.5 s |
| `0000019` | 59.3 s | 83.3 s |
| `0000052` | 107.0 s | 251.7 s |
| `0000350` | 72.5 s | 160.1 s |

EGF is slower than EG in this run, but the cost remains acceptable for confirmatory evaluation: 10 molecule total wall time was 33:30.03.

## Conclusion

The 10 molecule confirmatory run supports the 5 molecule result:

- The cost-reduced density-relaxed evaluator remains stable: 924/924 displacement points reached strict convergence.
- EGF lambda=1.0 remains clearly better than EG on all reported Hessian metrics.
- The EGF advantage is not confined to the original first 5 small molecules; it also holds on the added 5 molecule subset.
- The result can be described as a P1-410 early-pilot density-relaxed derived-force Hessian improvement trend.

This is still not a final P1/P2 conclusion. The sample is 10 molecules, and the Hessian is generated by finite difference of derived model force after density optimization.

## Next Step

Reasonable next step:

- Expand the same evaluator to 20 molecules using the same strict `base-density warm-start + two-stage Adam` setting, if the goal is to strengthen Hessian evidence before returning to larger label generation or broader training.

Do not jump to final claims or 1000 molecule labels solely from this 10 molecule result.
