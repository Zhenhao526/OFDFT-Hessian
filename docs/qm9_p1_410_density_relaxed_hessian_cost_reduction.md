# QM9 P1-410 Density-Relaxed Hessian Cost Reduction

Date: 2026-07-07

## Scope

This report summarizes cost-reduction experiments for the current 5 molecule P1-410 density-relaxed derived-force Hessian evaluator.

The evaluation definition is unchanged:

- For every finite-difference displaced geometry, run OFDFT density optimization first.
- Compute force from the scalar model energy only: `F_pred = -dE_pred/dR`.
- Build the Cartesian Hessian by finite difference of the derived force.
- No independent force head is used.
- No new labels are generated.

This is still a density-relaxed derived-force Hessian evaluation, not a full total OFDFT Hessian with analytic nuclear derivatives of all classical terms.

## Inputs

Dataset:

- `_runtime/qm9_p1/QM9PBEForcePilot`

Reference manifest:

- `_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json`

Only the first 5 reference molecules were used:

| molecule_id | natoms |
| --- | ---: |
| `0000010` | 6 |
| `0000323` | 6 |
| `0000109` | 7 |
| `0000171` | 7 |
| `0000062` | 8 |

Models:

| model | run dir | checkpoint |
| --- | --- | --- |
| EG_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200` | `checkpoints/epoch_006.ckpt` |
| EGF_lam1_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000` | `checkpoints/epoch_000.ckpt` |

## Evaluator Changes

Updated:

- `scripts/qm9_hessian_density_relaxed_eval.py`

Added optional controls:

- `--base-density-warm-start`
- `--fallback-always`

The base-density warm-start path first optimizes the original/base geometry density, then initializes each displaced geometry from the converged base coefficients transformed into the displaced sample basis with `transform_tensor_with_sample(..., Representation.VECTOR)`.

This is a model-density warm-start from the same molecule geometry neighborhood. It is not a new PBE label, not a label/reference density warm-start, and does not write back to labels.

## Complete Runs

Baseline strict run:

- Output prefix: `_runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_staged_adam1e-3_fb3e-4_restart2`
- Settings: `sad_default`, Adam `lr=1e-3`, `max_cycle=5000`, threshold `1e-4`; fallback Adam `lr=3e-4`, `max_cycle=10000`, threshold `1e-4` only when first stage failed.

Best cost-reduction run:

- Output prefix: `_runtime/qm9_p1_models/eval/qm9_p1_410_density_relaxed_hessian_cost_reduction/basewarm_twostage_strict`
- Settings:
  - Base density warm-start enabled.
  - Stage 1: Adam `lr=1e-3`, `max_cycle=1000`, threshold `1e-2`.
  - Stage 2: Adam `lr=3e-4`, `max_cycle=10000`, target threshold `1e-4`.
  - `--fallback-always`, with stage 2 skipped only if stage 1 is already below `1e-4`.

## Convergence And Cost

Displacement optimization points only; base warm-start pre-optimizations are reported separately below.

| scheme | status | target | opt points | strict `<1e-4` | medium `<1e-3` | mean cycles | median cycles | max cycles | fallback count | wall time | max RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict baseline | complete | `1e-4` | 408 | 408 | 408 | 1487.7 | 1357 | 5148 | 19 | 3:08:19 | 1.60 GB |
| basewarm + two-stage | complete | `1e-4` | 408 | 408 | 408 | 178.5 | 125 | 1114 | 408 | 17:01.99 | 1.60 GB |

Base warm-start overhead for `basewarm + two-stage`:

| model | base optimizations | mean base cycles | max base cycles | total base elapsed |
| --- | ---: | ---: | ---: | ---: |
| EG_s3000 | 5 | 713.2 | 897 | 40.0 s |
| EGF_lam1_s3000 | 5 | 922.0 | 974 | 60.3 s |
| all | 10 | 817.6 | 974 | 100.4 s |

Cost reduction versus strict baseline:

| model | strict elapsed | basewarm + two-stage elapsed | elapsed ratio | mean cycles ratio |
| --- | ---: | ---: | ---: | ---: |
| EG_s3000 | 4496.6 s | 439.6 s | 0.098 | 0.139 |
| EGF_lam1_s3000 | 6750.9 s | 540.2 s | 0.080 | 0.105 |
| combined wall | 3:08:19 | 17:01.99 | 0.090 | 0.120 |

The full 5 molecule strict run is reduced from about 3.1 hours to about 17 minutes while preserving 204/204 strict convergence per model.

## Hessian Metrics

Mean over the same 5 molecule reference set.

| scheme | model | Hessian MAE | Hessian RMSE | relative Fro | symmetry max abs |
| --- | --- | ---: | ---: | ---: | ---: |
| strict baseline | EG_s3000 | 1.804291 | 4.668794 | 23.497451 | 19.273678 |
| strict baseline | EGF_lam1_s3000 | 0.035395 | 0.087599 | 0.561274 | 0.175188 |
| basewarm + two-stage | EG_s3000 | 1.804253 | 4.668673 | 23.496900 | 19.273569 |
| basewarm + two-stage | EGF_lam1_s3000 | 0.035394 | 0.087596 | 0.561245 | 0.175163 |

Differences versus strict baseline:

| model | MAE delta | RMSE delta | relative Fro delta | symmetry delta |
| --- | ---: | ---: | ---: | ---: |
| EG_s3000 | -0.000038 | -0.000121 | -0.000551 | -0.000109 |
| EGF_lam1_s3000 | -0.000001 | -0.000004 | -0.000029 | -0.000025 |

The cost-reduced strict run reproduces the baseline Hessian metrics within numerical noise. The EG/EGF ordering is unchanged: EGF lambda=1.0 remains much better than EG on MAE, RMSE, relative Frobenius error, and symmetry error.

## Partial Runs

Three additional attempts ended with exit status 1 and did not write JSON/CSV outputs. Their logs contain completed molecule lines but no traceback. Treat them as interrupted/failed partial runs, not full validation.

| scheme | status | completed portion | convergence reported in log | partial elapsed from completed lines | process wall time | max RSS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| threshold `1e-3` | partial | EG 5/5 + EGF 1/5 | completed points 240/240 at target `1e-3` | EG 1909.2 s; EGF first molecule 649.7 s | 1:05:24 | 1.56 GB |
| two-stage, no base warm-start | partial | EG 5/5 + EGF 1/5 | completed points 240/240 at target `1e-4` | EG 1400.4 s; EGF first molecule 402.6 s | 45:22.19 | 1.36 GB |
| base warm-start, no forced two-stage | partial | EG 5/5 | completed points 204/204 at target `1e-4` | EG 750.9 s | 21:20.25 | 1.60 GB |

Partial EG-only comparisons:

| scheme | EG completed molecules | EG Hessian MAE mean | EG elapsed over completed molecules | note |
| --- | ---: | ---: | ---: | --- |
| strict baseline | 5 | 1.804291 | 4496.6 s | complete |
| threshold `1e-3` | 5 | 1.805176 | 1909.2 s | partial run; medium threshold only |
| two-stage, no base warm-start | 5 | 1.804280 | 1400.4 s | partial run |
| base warm-start, no forced two-stage | 5 | 1.804312 | 750.9 s | partial run |
| basewarm + two-stage | 5 | 1.804253 | 439.6 s | complete |

The partial runs support the same trend:

- Relaxing to `1e-3` reduces time and does not visibly perturb completed EG metrics, but EGF is incomplete.
- Two-stage Adam helps.
- Base warm-start helps substantially.
- Combining base warm-start with two-stage Adam is the only fully completed cost-reduction run and is also the fastest observed strategy.

## Answers To The Cost Questions

Which setting is fastest without changing the EG/EGF conclusion?

- Use `base-density warm-start + two-stage Adam`, still targeting strict `1e-4`.
- It completed 408/408 displacement optimizations at strict convergence.
- Wall time dropped from 3:08:19 to 17:01.99.
- Hessian metrics and EG/EGF ordering are unchanged within numerical noise.

Is threshold `1e-3` enough for larger-sample screening?

- Not fully established from this run, because the `1e-3` threshold experiment exited before finishing all EGF molecules and did not write JSON/CSV.
- Completed points were stable: EG 5/5 and EGF `0000010` matched strict metrics closely and preserved EGF advantage on the completed overlap.
- Recommendation: `1e-3` is reasonable as a cheap exploratory pre-screen only, but use the strict `1e-4` basewarm + two-stage setting for the next confirmatory 10 molecule run.

Does base-density warm-start significantly reduce cycles?

- Yes. In the complete combined strategy, mean displacement cycles dropped from 1487.7 to 178.5, an 8.3x reduction.
- Per-model elapsed ratios were 0.098 for EG and 0.080 for EGF.
- The base pre-optimization overhead was about 100 s total across 10 model/molecule base solves, much smaller than the saved displacement optimization time.

Is it safe to expand to 10 molecule?

- Yes, as a controlled next step using `base-density warm-start + two-stage Adam` with final threshold `1e-4`.
- Do not use this result to expand to 20/50 yet without first checking the 10 molecule run.
- Keep the current 5 molecule benchmark as the strict regression baseline.
- If `1e-3` is considered for screening, first rerun or complete the EGF side because the current `1e-3` evidence is partial.

## Recommended Next Command Shape

For the next 10 molecule validation, keep the same evaluator definition and use:

```bash
.venv/bin/python scripts/qm9_hessian_density_relaxed_eval.py \
  --manifest-json _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --run EG_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt \
  --run EGF_lam1_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt \
  --max-molecules 10 \
  --optimizer adam \
  --lr 1e-3 \
  --max-cycle 1000 \
  --convergence-tolerance 1e-2 \
  --fallback-optimizer adam \
  --fallback-lr 3e-4 \
  --fallback-max-cycle 10000 \
  --fallback-convergence-tolerance 1e-4 \
  --fallback-always \
  --base-density-warm-start \
  --initialization sad_default \
  --displacement 1e-3 \
  --device cuda:0
```

Keep the output prefix separate from the 5 molecule runs.

## Limitations

- The conclusion is limited to the P1-410 early pilot and this 5 molecule density-relaxed derived-force Hessian benchmark.
- The best cost-reduction strategy was validated on 5 molecules only.
- The threshold `1e-3` experiment is incomplete and should not be used as the final convergence setting without additional EGF confirmation.
- This does not justify generating 1000 molecule labels or changing the model architecture.
