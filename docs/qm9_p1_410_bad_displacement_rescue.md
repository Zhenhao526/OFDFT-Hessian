# QM9 P1-410 Bad Displacement Density Optimization Rescue

Date: 2026-07-06

## Scope

This audit reruns only the non-strict displacement density optimizations from the 5 molecule Adam5000 density-relaxed Hessian evaluation.

Previous run:

`_runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_adam5000_optimizations.csv`

Previous non-strict points:

| run | non-strict points | molecules |
|---|---:|---|
| EG_s3000 | 14 | `0000323`: 13, `0000109`: 1 |
| EGF_lam1_s3000 | 8 | `0000323`: 8 |
| total | 22 | 2 molecules |

This step does not generate new labels, does not delete existing data, and does not change the force definition. It only tests whether the previously non-strict density optimization points can be made strict-converged with more conservative optimizer settings.

## Script

New script:

`scripts/qm9_density_relaxed_bad_displacement_rescue.py`

Behavior:

- Reads previous per-displacement optimization CSV.
- Selects rows with `converged != True`.
- Rebuilds the same displaced geometry.
- Runs one or more optimizer variants.
- Records every optimization cycle to a curve CSV.
- Writes per-point and per-variant summaries.

For `label` initialization, the script uses the original sample's `of_labels/spatial/coeffs[-1]` and transforms it into the current displaced sample basis. This is a reference-density warm-start approximation for small displacements; it is not a new PBE label for the displaced geometry.

## Command

```bash
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
/usr/bin/time -v -o _runtime/qm9_p1_models/eval/qm9_p1_410_bad_displacement_rescue/rescue_bad22_time.txt \
.venv/bin/python scripts/qm9_density_relaxed_bad_displacement_rescue.py \
  --previous-optimization-csv _runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_adam5000_optimizations.csv \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --run EG_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt \
  --run EGF_lam1_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt \
  --variant sad_adam3e-4_10000:sad_default:adam:3e-4:10000 \
  --variant label_adam3e-4_10000:label:adam:3e-4:10000 \
  --variant label_adam3e-4_10000_slsqp10000:label:adam:3e-4:10000:slsqp:0:10000 \
  --output-json _runtime/qm9_p1_models/eval/qm9_p1_410_bad_displacement_rescue/rescue_bad22.json \
  --output-csv _runtime/qm9_p1_models/eval/qm9_p1_410_bad_displacement_rescue/rescue_bad22.csv \
  --summary-csv _runtime/qm9_p1_models/eval/qm9_p1_410_bad_displacement_rescue/rescue_bad22_summary.csv \
  --curves-csv _runtime/qm9_p1_models/eval/qm9_p1_410_bad_displacement_rescue/rescue_bad22_curves.csv \
  --device cuda:0
```

## Artifacts

| artifact | path | size / rows |
|---|---|---:|
| JSON | `_runtime/qm9_p1_models/eval/qm9_p1_410_bad_displacement_rescue/rescue_bad22.json` | 66 KB |
| per-point CSV | `_runtime/qm9_p1_models/eval/qm9_p1_410_bad_displacement_rescue/rescue_bad22.csv` | 67 rows including header |
| summary CSV | `_runtime/qm9_p1_models/eval/qm9_p1_410_bad_displacement_rescue/rescue_bad22_summary.csv` | 7 rows including header |
| full curve CSV | `_runtime/qm9_p1_models/eval/qm9_p1_410_bad_displacement_rescue/rescue_bad22_curves.csv` | 223,497 rows including header, 41 MB |
| time log | `_runtime/qm9_p1_models/eval/qm9_p1_410_bad_displacement_rescue/rescue_bad22_time.txt` | 2.0 KB |

Resource usage:

| metric | value |
|---|---:|
| wall time | 1:05:53 |
| user time | 4258.11 s |
| system time | 44.91 s |
| CPU utilization | 108% |
| max RSS | 1,691,640 KB |
| device | `CUDA_VISIBLE_DEVICES=1`, script device `cuda:0` |

## Results

Strict convergence threshold: projected density-gradient norm `< 1e-4`.

| run | variant | converged | mean final grad | max final grad | mean cycles | cycle range | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|
| EG_s3000 | `sad_adam3e-4_10000` | 14/14 | 8.630e-05 | 9.631e-05 | 3505.0 | 3382-3658 | 13.60 min |
| EG_s3000 | `label_adam3e-4_10000` | 14/14 | 9.105e-05 | 9.999e-05 | 3268.7 | 3198-3430 | 12.65 min |
| EG_s3000 | `label_adam3e-4_10000_slsqp10000` | 14/14 | 8.868e-05 | 9.911e-05 | 3274.4 | 3181-3534 | 12.65 min |
| EGF_lam1_s3000 | `sad_adam3e-4_10000` | 8/8 | 9.280e-05 | 9.915e-05 | 3791.5 | 3746-3947 | 9.74 min |
| EGF_lam1_s3000 | `label_adam3e-4_10000` | 8/8 | 9.542e-05 | 9.996e-05 | 3274.3 | 3232-3349 | 8.29 min |
| EGF_lam1_s3000 | `label_adam3e-4_10000_slsqp10000` | 8/8 | 9.557e-05 | 9.923e-05 | 3287.0 | 3258-3399 | 8.30 min |

All 66 rescue runs converged. All recorded curve statuses were `decreasing`.

## Two-Stage Optimizer Check

The two-stage variant was configured as:

`label warm-start -> Adam(lr=3e-4, max_cycle=10000) -> SLSQP(max_cycle=10000)`

Observed result:

- `n_stages_run = 1` for all 22 two-stage runs.
- Adam reached strict convergence before SLSQP was entered.
- Therefore SLSQP was not needed for these 22 points.

The effective rescue setting is simply:

`Adam lr=3e-4, max_cycle=10000`

`label` warm-start is useful for speed, especially for EGF, but not required for convergence in this 22 point audit.

## Near-Threshold Points

Several successful points ended close to the strict threshold. These are converged by the current criterion but have little margin:

| run | molecule | coord | side | variant | final grad |
|---|---|---:|---|---|---:|
| EG_s3000 | `0000323` | 14 | plus | `label_adam3e-4_10000` | 9.999e-05 |
| EG_s3000 | `0000109` | 8 | minus | `label_adam3e-4_10000_slsqp10000` | 9.911e-05 |
| EGF_lam1_s3000 | `0000323` | 5 | minus | `label_adam3e-4_10000` | 9.906e-05 |
| EGF_lam1_s3000 | `0000323` | 5 | minus | `label_adam3e-4_10000_slsqp10000` | 9.923e-05 |
| EGF_lam1_s3000 | `0000323` | 13 | plus | `label_adam3e-4_10000_slsqp10000` | 9.913e-05 |
| EGF_lam1_s3000 | `0000323` | 15 | plus | `label_adam3e-4_10000` | 9.955e-05 |
| EGF_lam1_s3000 | `0000323` | 15 | minus | `sad_adam3e-4_10000` | 9.915e-05 |
| EGF_lam1_s3000 | `0000323` | 17 | plus | `label_adam3e-4_10000` | 9.996e-05 |

For a stricter margin, one could continue a subset to `<5e-5`, but this was not required by the current strict threshold.

## Interpretation

The previous non-strict points were not fundamentally blocked. Lowering Adam lr from `1e-3` to `3e-4` and allowing up to 10000 cycles resolves all 22 originally non-strict displacement density optimizations.

Main findings:

- `max_cycle=5000` was not the only issue; the optimizer path at `lr=1e-3` was too aggressive for these displacement points.
- `Adam lr=3e-4` is the stable setting for these bad points.
- `label` warm-start modestly reduces cycles and wall time, especially for EGF, but `sad_default` also converges all points.
- The second-stage SLSQP fallback was not triggered because first-stage Adam converged every point.

## Impact On Hessian Eval

This rescue run does not yet update the density-relaxed Hessian metrics, because it records density optimization convergence curves only. It does not recompute and splice the derived forces back into the finite-difference Hessian matrix.

Supported conclusion:

> The remaining density optimization convergence blocker in the 5 molecule P1-410 density-relaxed Hessian audit is mostly removed for the previously bad displacement points by Adam lr=3e-4, max_cycle=10000.

Not yet supported:

> The 5 molecule Hessian metrics have been recomputed with all rescued displacement forces.

Recommended next step:

1. Rerun the 5 molecule density-relaxed Hessian evaluator using `Adam lr=3e-4`, `max_cycle=10000`, and optionally `label` warm-start if the evaluator is extended to support it.
2. Confirm all 204 displacement optimizations are strict-converged.
3. Recompute EG vs EGF Hessian MAE/RMSE/relative Frobenius using the fully strict-converged force matrix.
4. Only then decide whether to expand to 10 molecules.

