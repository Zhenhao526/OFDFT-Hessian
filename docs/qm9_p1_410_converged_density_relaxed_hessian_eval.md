# QM9 P1-410 Converged Density-Relaxed Hessian Eval

Date: 2026-07-03

## Scope

This report extends the P1-410 early pilot Hessian audit from fixed-density / single-SCF proxies to a density-relaxed derived-force finite-difference evaluation on the same first 5 molecules from the PBE Hessian reference set.

Constraints kept:

- No 1000 molecule label generation.
- No deletion of existing labels, checkpoints, cached labels, or Hessian references.
- No independent force head.
- Predicted force remains derived from scalar model energy: `F_pred = -dE_pred/dR`.
- Conclusions are limited to the P1-410 early pilot.

## Evaluation Definition

Current evaluator: `scripts/qm9_hessian_density_relaxed_eval.py`

For each molecule and each Cartesian finite-difference displacement:

1. Build the displaced PySCF molecule.
2. Generate an OFDFT sample from the model run configuration.
3. Optimize density with:
   - initialization: `sad_default`
   - optimizer: Adam
   - lr: `1e-3`
   - max cycle: `5000`
   - strict threshold: projected density-gradient norm `< 1e-4`
4. Re-enable coordinate gradients after density optimization.
5. Compute force through the model scalar energy autograd path.
6. Build the Hessian by central finite difference of derived force.

This should be called a density-relaxed derived-force finite-difference Hessian proxy, not a full total OFDFT Hessian. The density is optimized at each displaced geometry, but the reported force path differentiates the model scalar energy and does not include explicit nuclear derivatives of the classical integral / nuclear terms.

## Inputs

Reference manifest:

`_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json`

First 5 references used:

| molecule | sample_id | natoms |
|---|---:|---:|
| `0000010` | 0 | 6 |
| `0000323` | 0 | 6 |
| `0000109` | 0 | 7 |
| `0000171` | 0 | 7 |
| `0000062` | 0 | 8 |

Checkpoints:

| run | checkpoint |
|---|---|
| EG_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt` |
| EGF_lam1_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt` |

Outputs:

| artifact | path |
|---|---|
| JSON | `_runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_adam5000.json` |
| molecule CSV | `_runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_adam5000.csv` |
| per-displacement optimization CSV | `_runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_adam5000_optimizations.csv` |
| `/usr/bin/time -v` | `_runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_adam5000_time.txt` |

Command:

```bash
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
/usr/bin/time -v -o _runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_adam5000_time.txt \
.venv/bin/python scripts/qm9_hessian_density_relaxed_eval.py \
  --manifest-json _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --run EG_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt \
  --run EGF_lam1_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt \
  --output-json _runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_adam5000.json \
  --output-csv _runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_adam5000.csv \
  --optimization-csv _runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_adam5000_optimizations.csv \
  --max-molecules 5 \
  --optimizer adam \
  --max-cycle 5000 \
  --convergence-tolerance 1e-4 \
  --lr 1e-3 \
  --displacement 1e-3 \
  --device cuda:0 \
  --initialization sad_default
```

## Density Optimization Convergence

Strict convergence means final projected density-gradient norm `< 1e-4`.

| run | strict converged displacement points | Hessian finite force points | mean final grad norm | max final grad norm | max cycles |
|---|---:|---:|---:|---:|---:|
| EG_s3000 | 190/204 (93.1%) | 204/204 | 3.146e-04 | 7.237e-03 | 5000 |
| EGF_lam1_s3000 | 196/204 (96.1%) | 204/204 | 1.875e-04 | 4.732e-03 | 5000 |

Non-strict displacement points:

| run | non-strict count | molecules |
|---|---:|---|
| EG_s3000 | 14 | `0000323`: 13, `0000109`: 1 |
| EGF_lam1_s3000 | 8 | `0000323`: 8 |

There were no molecule-level failures. All force evaluations were finite.

Resource usage:

| metric | value |
|---|---:|
| wall time | 1:44:00 |
| user time | 9408.44 s |
| system time | 171.04 s |
| CPU utilization | 153% |
| max RSS | 1,597,112 KB |
| device | `CUDA_VISIBLE_DEVICES=1`, script device `cuda:0` |

## Hessian Metrics vs PBE Reference

Mean over 5 molecules:

| run | strict convergence | Hessian MAE | Hessian RMSE | relative Frobenius | symmetry max abs | total eval time |
|---|---:|---:|---:|---:|---:|---:|
| EG_s3000 | 190/204 (93.1%) | 1.804279 | 4.668802 | 23.497507 | 19.273474 | 40.7 min |
| EGF_lam1_s3000 | 196/204 (96.1%) | 0.035395 | 0.087599 | 0.561268 | 0.175188 | 62.7 min |

EGF vs EG reductions:

| metric | reduction |
|---|---:|
| Hessian MAE | 98.0% lower |
| Hessian RMSE | 98.1% lower |
| relative Frobenius | 97.6% lower |
| symmetry max abs | 99.1% lower |

Per molecule:

| molecule | natoms | EG conv | EG MAE | EG RMSE | EG rel Fro | EGF conv | EGF MAE | EGF RMSE | EGF rel Fro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0000010` | 6 | 36/36 | 1.073275 | 5.559395 | 32.670555 | 36/36 | 0.047405 | 0.112143 | 0.659023 |
| `0000323` | 6 | 23/36 | 5.888114 | 11.607356 | 50.077553 | 28/36 | 0.046211 | 0.095036 | 0.410014 |
| `0000109` | 7 | 41/42 | 1.742733 | 5.076758 | 26.115271 | 42/42 | 0.020994 | 0.050930 | 0.261990 |
| `0000171` | 7 | 42/42 | 0.105006 | 0.347618 | 2.936643 | 42/42 | 0.044959 | 0.130278 | 1.100576 |
| `0000062` | 8 | 48/48 | 0.212269 | 0.752881 | 5.687511 | 48/48 | 0.017409 | 0.049606 | 0.374738 |

## Comparison With Previous Proxy Evaluations

Same first 5 molecules:

| evaluator | run | convergence | Hessian MAE | Hessian RMSE | relative Frobenius | symmetry max abs |
|---|---|---:|---:|---:|---:|---:|
| fixed-density / single-SCF proxy | EG_s3000 | n/a | 0.435238 | 1.856111 | 10.087106 | 0.003304 |
| fixed-density / single-SCF proxy | EGF_lam1_s3000 | n/a | 0.021418 | 0.058155 | 0.387190 | 0.000098 |
| density-relaxed, SGD50, unconverged | EG_s3000 | 0/204 | 0.727102 | 2.277429 | 12.198326 | 8.620611 |
| density-relaxed, SGD50, unconverged | EGF_lam1_s3000 | 0/204 | 0.034219 | 0.081952 | 0.524189 | 0.165261 |
| density-relaxed, Adam5000 | EG_s3000 | 190/204 | 1.804279 | 4.668802 | 23.497507 | 19.273474 |
| density-relaxed, Adam5000 | EGF_lam1_s3000 | 196/204 | 0.035395 | 0.087599 | 0.561268 | 0.175188 |
| density-relaxed, Adam5000 + fallback Adam3e-4 | EG_s3000 | 204/204 | 1.804291 | 4.668794 | 23.497451 | 19.273678 |
| density-relaxed, Adam5000 + fallback Adam3e-4 | EGF_lam1_s3000 | 204/204 | 0.035395 | 0.087599 | 0.561274 | 0.175188 |

Interpretation:

- The EG/EGF ordering is stable across fixed-density, unconverged density-relaxed, and Adam5000 density-relaxed evaluators.
- Density relaxation changes absolute Hessian errors, especially for EG, and can expose severe asymmetry in derived-force finite differences.
- EGF remains much closer to PBE Hessian reference on this 5 molecule set.
- Adam5000 substantially improves convergence over SGD50, but does not make every displacement point strict-converged.
- Adding a second-stage Adam fallback at lr `3e-4` for only first-stage failures makes all 204 displacement optimizations strict-converged.

## Staged Fallback Rerun

Date: 2026-07-06

Command variant:

- First stage: `sad_default`, Adam lr `1e-3`, max cycle `5000`, threshold `1e-4`.
- Fallback stage: only if first stage fails, Adam lr `3e-4`, max cycle `10000`, threshold `1e-4`, starting from the first-stage final density coefficients.

Artifacts:

| artifact | path |
|---|---|
| JSON | `_runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_staged_adam1e-3_fb3e-4_restart2.json` |
| molecule CSV | `_runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_staged_adam1e-3_fb3e-4_restart2.csv` |
| per-displacement CSV | `_runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_staged_adam1e-3_fb3e-4_restart2_optimizations.csv` |
| time log | `_runtime/qm9_p1_models/eval/qm9_p1_410_converged_density_relaxed_hessian_eval/density_relaxed_5mol_staged_adam1e-3_fb3e-4_restart2_time.txt` |

Final mean metrics:

| run | strict convergence | Hessian MAE | Hessian RMSE | relative Frobenius | symmetry max abs | total eval time |
|---|---:|---:|---:|---:|---:|---:|
| EG_s3000 | 204/204 | 1.804291 | 4.668794 | 23.497451 | 19.273678 | 74.9 min |
| EGF_lam1_s3000 | 204/204 | 0.035395 | 0.087599 | 0.561274 | 0.175188 | 112.5 min |

Per molecule:

| molecule | natoms | EG conv | EG MAE | EG RMSE | EG rel Fro | EGF conv | EGF MAE | EGF RMSE | EGF rel Fro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0000010` | 6 | 36/36 | 1.073284 | 5.559384 | 32.670489 | 36/36 | 0.047405 | 0.112144 | 0.659032 |
| `0000323` | 6 | 36/36 | 5.888110 | 11.607349 | 50.077519 | 36/36 | 0.046207 | 0.095036 | 0.410013 |
| `0000109` | 7 | 42/42 | 1.742793 | 5.076762 | 26.115292 | 42/42 | 0.020996 | 0.050931 | 0.261995 |
| `0000171` | 7 | 42/42 | 0.104998 | 0.347595 | 2.936446 | 42/42 | 0.044958 | 0.130281 | 1.100599 |
| `0000062` | 8 | 48/48 | 0.212269 | 0.752881 | 5.687511 | 48/48 | 0.017410 | 0.049605 | 0.374731 |

Fallback trigger summary:

| run | total displacement points | first-stage converged | fallback used | fallback converged | fallback molecules |
|---|---:|---:|---:|---:|---|
| EG_s3000 | 204 | 195 | 9 | 9 | `0000323` |
| EGF_lam1_s3000 | 204 | 194 | 10 | 10 | `0000323` |

Resource usage:

| metric | value |
|---|---:|
| wall time | 3:08:19 |
| user time | 13169.03 s |
| system time | 136.91 s |
| CPU utilization | 117% |
| max RSS | 1,602,632 KB |

## Conclusion

EGF lambda=1.0 s3000 still shows a strong Hessian improvement trend after density optimization is inserted at every displaced geometry. With staged fallback, all 204 displacement optimizations are strict-converged for both EG and EGF. On this 5 molecule audit, EGF has lower Hessian MAE/RMSE/relative Frobenius on every molecule and improves the 5-molecule average by roughly 98% in MAE and RMSE relative to EG.

Remaining limitations:

- The evaluated Hessian is still a derived-force finite-difference proxy with density relaxation, not a full total OFDFT Hessian including all explicit nuclear derivative terms.
- The reference set is still only 5 small molecules from the P1-410 early pilot.
- The conclusion should not be extrapolated to final P1/P2.

Therefore the supported statement is:

> In the P1-410 early pilot, EGF lambda=1.0 s3000 shows a robust 5 molecule density-relaxed derived-force Hessian improvement over EG when first-stage failures are completed with Adam lr=3e-4 fallback.

This can be called a strict-converged density-relaxed derived-force Hessian result for the 5 molecule audit. It still should not be called a full physical OFDFT Hessian benchmark.

## Next Decision

The 5 molecule blocker is cleared for this evaluator: all displacement points are strict-converged using first-stage Adam lr `1e-3` plus fallback Adam lr `3e-4`.

Recommended next step:

1. Expand the same staged fallback evaluator to 10 molecules, not 20/50 yet.
2. Keep the existing 5 molecule result as the strict-converged density-relaxed baseline.
3. Track fallback trigger counts and wall time before deciding whether a 20 molecule rerun is affordable.
