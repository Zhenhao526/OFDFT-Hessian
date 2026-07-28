# QM9 P1-410 Fixed-Density vs Density-Relaxed Hessian Time Comparison

Date: 2026-07-07

## Scope

This report compares three Hessian objects on the existing 10 molecule PBE Hessian reference set:

1. KSDFT/PBE analytic Hessian reference.
2. OFDFT fixed-density Hessian proxy.
3. OFDFT density-relaxed derived-force Hessian.

Constraints kept:

- no new labels generated;
- no existing labels, cached labels, checkpoints, or Hessian references deleted;
- no independent force head;
- force and Hessian are derived from scalar model energy;
- fixed-density Hessian remains a proxy and is not treated as the final physical Hessian.

The conclusion is limited to the P1-410 early pilot.

## Inputs

Molecules:

```text
0000010, 0000323, 0000109, 0000171, 0000062,
0000116, 0000144, 0000019, 0000052, 0000350
```

Models:

| model | checkpoint |
| --- | --- |
| EG_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt` |
| EGF_lam1_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt` |

Artifacts:

| artifact | path |
| --- | --- |
| fixed-density autograd / FD Hessian | `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_10mol_validation/second_order_autograd_10mol.json` |
| fixed-density Hessian matrices | `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_10mol_validation/hessians/` |
| density-relaxed matrix rerun | `_runtime/qm9_p1_models/eval/qm9_p1_410_fixed_vs_density_relaxed_hessian_time_comparison/density_relaxed_with_matrices.json` |
| density-relaxed Hessian matrices | `_runtime/qm9_p1_models/eval/qm9_p1_410_fixed_vs_density_relaxed_hessian_time_comparison/density_relaxed_hessians/` |
| PBE 10 molecule manifest | `_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessian_reference_manifest.json` |

The density-relaxed evaluator was updated with an optional `--hessian-npz-dir` output so the density-relaxed Hessian matrices can be compared directly against fixed-density matrices. This only adds saved evaluation artifacts; it does not modify labels or checkpoints.

## Definitions

PBE reference:

- Existing KSDFT/PBE analytic Hessian reference from the 10 molecule manifest.

Fixed-density OFDFT Hessian:

- Full-edge second-order autograd Hessian from scalar model energy.
- Density coefficients are fixed at `scf_iteration=1`.
- No density optimization is run after coordinate changes.
- This is a single-SCF / fixed-coefficient proxy.
- Fixed-density FD Hessian is included only as a consistency check against autograd.

Density-relaxed OFDFT Hessian:

- Cartesian Hessian is built by finite difference of derived force.
- For every displaced geometry, density optimization is run first.
- Force is still `F_pred = -dE_pred/dR` from scalar model energy.
- Settings: `sad_default`, base-density warm-start, Adam `lr=1e-3` to threshold `1e-2` or 1000 cycles, then Adam `lr=3e-4` to threshold `1e-4` with `fallback-always`.
- This is a density-relaxed derived-force finite-difference Hessian, not an analytic implicit Hessian implementation.

## Aggregate Hessian Comparison

Mean over the 10 molecule set:

| model | fixed auto vs PBE MAE | fixed auto vs PBE RMSE | fixed auto vs PBE rel Fro | fixed auto sym max | density-relaxed vs PBE MAE | density-relaxed vs PBE RMSE | density-relaxed vs PBE rel Fro | density-relaxed sym max | fixed auto vs relaxed MAE | fixed auto vs relaxed RMSE | fixed auto vs relaxed rel Fro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EG_s3000 | 0.418686 | 1.976556 | 12.802920 | 4.250e-07 | 1.192575 | 3.588603 | 21.397202 | 13.721052 | 0.836616 | 2.368275 | 0.506211 |
| EGF_lam1_s3000 | 0.019494 | 0.054171 | 0.392696 | 2.281e-08 | 0.041653 | 0.106805 | 0.834249 | 0.414682 | 0.026179 | 0.072212 | 0.327924 |

Fixed-density FD agrees with fixed-density autograd:

| model | fixed FD vs PBE MAE | fixed FD vs PBE RMSE | fixed FD vs PBE rel Fro | fixed FD sym max | fixed FD vs relaxed MAE | fixed FD vs relaxed RMSE | fixed FD vs relaxed rel Fro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EG_s3000 | 0.418443 | 1.975134 | 12.794037 | 0.002771 | 0.836603 | 2.368335 | 0.506358 |
| EGF_lam1_s3000 | 0.019494 | 0.054170 | 0.392688 | 5.466e-05 | 0.026179 | 0.072212 | 0.327919 |

Interpretation:

- Fixed-density autograd and fixed-density FD are effectively the same for this comparison.
- Fixed-density and density-relaxed Hessians are not matrix-equivalent.
- The gap is large for EG and smaller but still visible for EGF.
- Density relaxation does not make these checkpoints closer to PBE on this 10 molecule set. Compared with fixed-density autograd, density-relaxed MAE is worse on 9/10 EG molecules and 10/10 EGF molecules.
- EGF lambda=1.0 is still much better than EG under both fixed-density and density-relaxed definitions.

## Per-Molecule Hessian Metrics

| molecule | natoms | model | fixed auto MAE vs PBE | relaxed MAE vs PBE | fixed-vs-relaxed MAE | fixed auto rel Fro vs PBE | relaxed rel Fro vs PBE | fixed-vs-relaxed rel Fro | fixed auto s | relaxed s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0000010` | 6 | EG_s3000 | 0.500493 | 1.073280 | 0.703955 | 21.171921 | 32.669953 | 0.798015 | 0.799 | 66.4 |
| `0000010` | 6 | EGF_lam1_s3000 | 0.011799 | 0.047405 | 0.038773 | 0.140195 | 0.659013 | 0.594031 | 0.186 | 92.0 |
| `0000323` | 6 | EG_s3000 | 0.917553 | 5.888024 | 5.025690 | 11.202514 | 50.076223 | 0.893832 | 0.425 | 127.7 |
| `0000323` | 6 | EGF_lam1_s3000 | 0.024848 | 0.046204 | 0.025804 | 0.306020 | 0.409985 | 0.270982 | 0.181 | 162.3 |
| `0000109` | 7 | EG_s3000 | 0.508172 | 1.742668 | 1.321677 | 11.798892 | 26.114073 | 0.763377 | 0.803 | 66.2 |
| `0000109` | 7 | EGF_lam1_s3000 | 0.014568 | 0.020991 | 0.008446 | 0.194065 | 0.261973 | 0.130333 | 0.211 | 107.6 |
| `0000171` | 7 | EG_s3000 | 0.101656 | 0.105003 | 0.011451 | 2.801783 | 2.936460 | 0.117540 | 0.569 | 52.1 |
| `0000171` | 7 | EGF_lam1_s3000 | 0.039484 | 0.044957 | 0.008156 | 0.946838 | 1.100568 | 0.384591 | 0.213 | 71.3 |
| `0000062` | 8 | EG_s3000 | 0.150044 | 0.212291 | 0.112397 | 3.501976 | 5.688057 | 0.600495 | 0.378 | 57.8 |
| `0000062` | 8 | EGF_lam1_s3000 | 0.016382 | 0.017410 | 0.002255 | 0.348770 | 0.374721 | 0.058410 | 0.239 | 77.1 |
| `0000116` | 8 | EG_s3000 | 0.337313 | 0.493951 | 0.340696 | 15.095424 | 14.146207 | 0.622613 | 0.489 | 84.3 |
| `0000116` | 8 | EGF_lam1_s3000 | 0.014531 | 0.027462 | 0.018284 | 0.368618 | 0.480766 | 0.435028 | 0.239 | 128.0 |
| `0000144` | 8 | EG_s3000 | 0.316969 | 0.340867 | 0.039535 | 17.901942 | 19.353224 | 0.075916 | 0.241 | 59.9 |
| `0000144` | 8 | EGF_lam1_s3000 | 0.019703 | 0.020810 | 0.005070 | 0.319427 | 0.332786 | 0.168429 | 0.239 | 77.7 |
| `0000019` | 9 | EG_s3000 | 0.352306 | 0.340530 | 0.039325 | 13.475336 | 12.389234 | 0.094463 | 0.431 | 61.4 |
| `0000019` | 9 | EGF_lam1_s3000 | 0.015269 | 0.015714 | 0.003600 | 0.395403 | 0.387956 | 0.093152 | 0.271 | 85.2 |
| `0000052` | 9 | EG_s3000 | 0.083679 | 0.716653 | 0.670570 | 2.836119 | 19.395769 | 0.991060 | 0.616 | 111.9 |
| `0000052` | 9 | EGF_lam1_s3000 | 0.019421 | 0.155933 | 0.146344 | 0.581734 | 3.984976 | 1.014180 | 0.271 | 179.0 |
| `0000350` | 9 | EG_s3000 | 0.918679 | 1.012484 | 0.100868 | 28.243289 | 31.202819 | 0.104795 | 0.277 | 74.5 |
| `0000350` | 9 | EGF_lam1_s3000 | 0.018935 | 0.019644 | 0.005061 | 0.325889 | 0.349749 | 0.130102 | 0.272 | 102.7 |

## Ranking Consistency

EGF lambda=1.0 is better than EG on Hessian MAE for every molecule:

| comparison | EGF better molecules |
| --- | ---: |
| fixed-density autograd vs PBE | 10/10 |
| fixed-density FD vs PBE | 10/10 |
| density-relaxed vs PBE | 10/10 |

Thus the EG/EGF ordering is stable across fixed-density proxy and density-relaxed evaluation. The absolute Hessian matrices still differ between fixed-density and density-relaxed paths, so ranking consistency should not be interpreted as matrix equivalence.

## Time and Resource Cost

Aggregate timing:

| scope | wall / sum time | max RSS / peak | note |
| --- | ---: | ---: | --- |
| PBE reference set | 53:54.87 wall; 6324.8 s manifest sum; 6324.8 s uncached sum | 2.18 GB | workers=2; `0000010` was cached |
| fixed-density autograd validation | 1:01.60 wall; 7.35 s autograd row sum; 8.57 s FD row sum | 1.75 GB RSS; 195.5 MB CUDA peak | both models; script also ran FD/HVP |
| density-relaxed matrix rerun | 31:13.83 wall; 1845.2 s row elapsed sum | 1.74 GB RSS | both models, strict threshold `1e-4` |

Speed summary:

- Fixed-density autograd validation wall time is about 30.4x faster than the density-relaxed matrix rerun wall time.
- Comparing row sums, fixed-density autograd Hessian is about 251x faster than density-relaxed Hessian for the two-model 10 molecule set.
- Density-relaxed OFDFT, evaluating both EG and EGF, is about 1.73x faster than the PBE reference wall time in this small run.
- Model-specific density-relaxed row-time comparison against PBE wall gives about 4.24x for EG and 2.99x for EGF.
- These PBE speedups are not fully fair because PBE used a cached `0000010`, two workers, and a different analytic Hessian code path. They are useful as operational cost estimates, not as a final benchmark.

Per-molecule timing, both models combined:

| molecule | natoms | PBE reference s | PBE s/atom | fixed autograd both models s | fixed autograd s/atom | density-relaxed both models s | density-relaxed s/atom | relaxed/autograd speedup | PBE/density-relaxed speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0000010` | 6 | 0.0 cached | 0.0 | 0.985 | 0.164 | 158.4 | 26.4 | 160.8x | 0.00x |
| `0000323` | 6 | 526.2 | 87.7 | 0.606 | 0.101 | 290.0 | 48.3 | 478.5x | 1.81x |
| `0000109` | 7 | 544.1 | 77.7 | 1.014 | 0.145 | 173.8 | 24.8 | 171.4x | 3.13x |
| `0000171` | 7 | 564.2 | 80.6 | 0.782 | 0.112 | 123.4 | 17.6 | 157.8x | 4.57x |
| `0000062` | 8 | 668.0 | 83.5 | 0.618 | 0.077 | 134.9 | 16.9 | 218.4x | 4.95x |
| `0000116` | 8 | 607.3 | 75.9 | 0.728 | 0.091 | 212.3 | 26.5 | 291.6x | 2.86x |
| `0000144` | 8 | 725.1 | 90.6 | 0.481 | 0.060 | 137.6 | 17.2 | 286.3x | 5.27x |
| `0000019` | 9 | 612.2 | 68.0 | 0.701 | 0.078 | 146.7 | 16.3 | 209.2x | 4.17x |
| `0000052` | 9 | 926.9 | 103.0 | 0.887 | 0.099 | 290.9 | 32.3 | 328.0x | 3.19x |
| `0000350` | 9 | 1150.7 | 127.9 | 0.549 | 0.061 | 177.2 | 19.7 | 322.7x | 6.49x |

Convergence for the density-relaxed matrix rerun:

- 924/924 displaced density optimizations reached strict threshold `<1e-4`.
- No failed molecule in this 10 molecule rerun.

## Hard Cases

`0000323`:

- Hardest EG density-relaxed case.
- EG relaxed MAE vs PBE is `5.888024`, relative Frobenius error is `50.076223`.
- Fixed-vs-relaxed EG MAE is `5.025690`.
- EGF remains controlled on the same molecule: relaxed MAE `0.046204`.

`0000052`:

- Main EGF density-relaxed hard case.
- EGF fixed-density MAE vs PBE is `0.019421`, but density-relaxed MAE rises to `0.155933`.
- EGF fixed-vs-relaxed relative Frobenius error is `1.014180`.
- This molecule should be kept as a required regression case for any future density-relaxed or implicit-Hessian implementation.

`0000010`:

- PBE timing is cached and cannot be used for a fair speed comparison.
- EG density-relaxed symmetry max abs error is large in the 10 molecule benchmark.

`0000350`:

- Largest PBE reference time in this set: `1150.7 s`.
- EG Hessian remains poor, while EGF remains close to PBE.

## Answers to the Main Questions

1. Is fixed-density Hessian a fast proxy for density-relaxed Hessian?

Yes for fast screening and EG/EGF ranking on this set, but no as a matrix-accurate replacement. It is much faster and preserves the EGF-over-EG ordering on 10/10 molecules. However, fixed-vs-relaxed errors are not negligible, especially for EG and for `0000052` under EGF.

2. Does density relaxation make Hessian closer to PBE?

Not for these checkpoints on this 10 molecule set. Relative to fixed-density autograd, density-relaxed MAE vs PBE is worse on 9/10 EG molecules and 10/10 EGF molecules. This may reflect model/density optimization error or cancellation in the fixed-density proxy; it does not make fixed-density a physical substitute.

3. Does EGF lambda=1.0 remain better under both definitions?

Yes. EGF lambda=1.0 is better than EG on fixed-density autograd, fixed-density FD, and density-relaxed Hessian MAE for all 10 molecules.

4. Which molecules are hard cases?

The main hard cases are `0000323` for EG density-relaxed Hessian and `0000052` for EGF density-relaxed deviation from fixed-density. `0000350` is the slowest PBE reference, and `0000010` has cached PBE time so it should not be used for cost conclusions.

5. Is fixed-density autograd suitable for fast screening?

Yes. It is stable, finite, highly consistent with fixed-density FD, and much faster than density-relaxed FD. It is appropriate for screening checkpoints, molecules, and likely outliers before paying density-relaxed cost. It should remain labeled as a proxy.

6. Does OFDFT Hessian evaluation have practical speed advantage over KSDFT/PBE reference?

Fixed-density autograd clearly has a large speed advantage. Density-relaxed OFDFT also has an operational speed advantage in this run, but the comparison is not fully controlled. The two-model density-relaxed run took `31:13.83`, while the PBE reference run took `53:54.87`; model-specific density-relaxed row sums are faster than PBE wall by about `3x-4x`. A fair final speed claim needs identical hardware, uncached PBE reference timings, and a larger molecule set.

## Conclusion

For P1-410 early pilot:

- Fixed-density second-order autograd Hessian is a useful high-throughput proxy and screening metric.
- It is not equivalent to the density-relaxed Hessian matrix.
- Density relaxation changes the Hessian substantially and, for these checkpoints, does not improve closeness to PBE on the 10 molecule set.
- EGF lambda=1.0 remains consistently better than EG under both fixed-density and density-relaxed Hessian evaluations.
- The official physical-style conclusion should still rely on density-relaxed FD Hessian, while fixed-density autograd can be used to prioritize what to evaluate next.

Recommended next step:

- Keep fixed-density autograd as the cheap screening path.
- Keep `0000323` and `0000052` as mandatory hard-case regression molecules.
- Do not replace density-relaxed FD conclusions with fixed-density autograd until a broader fixed-vs-relaxed validation is completed.
