# QM9 P1-410 Density Optimization Convergence Sweep

Date: 2026-07-03

Scope: QM9PBEForcePilot P1-410 early pilot only. This does not expand Hessian references and does not generate 1000 molecule labels.

## Goal

The previous 5 molecule physical Hessian audit showed that both EG and EGF lambda=1.0 failed density optimization convergence on the original `sample_id=0` geometries under:

- `sad_default`
- SGD, `lr=1e-3`, `momentum=0.9`
- `max_cycle=50`
- strict threshold `1e-4`

This sweep tests whether the failure is caused by too few cycles, bad learning rate, optimizer choice, or initialization.

## Inputs

Molecules:

- `0000010`
- `0000323`
- `0000109`
- `0000171`
- `0000062`

Dataset:

- `_runtime/qm9_p1/QM9PBEForcePilot`

Checkpoints:

| Run | Run dir | Checkpoint |
|---|---|---|
| EG_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200` | `checkpoints/epoch_006.ckpt` |
| EGF_lam1_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000` | `checkpoints/epoch_000.ckpt` |

## Sweep Design

Script added:

- `scripts/qm9_density_optimization_convergence_sweep.py`

The script runs each `(model, molecule, optimizer, lr, initialization)` once to the largest max-cycle budget using the strictest threshold, then derives the shorter max-cycle and relaxed-threshold results from the saved gradient norm curve.

This avoids physically repeating identical trajectories for each threshold.

Sweep grid:

| Axis | Values |
|---|---|
| Models | EG_s3000, EGF_lam1_s3000 |
| Molecules | 5 listed above |
| Optimizer | SGD, Adam |
| Learning rate | `1e-3`, `3e-4`, `1e-4` |
| Initialization | `sad_default`, `label` |
| Max-cycle budgets derived | 50, 100, 200, 500 |
| Thresholds derived | `1e-2`, `1e-3`, `1e-4` |

Notes:

- Adam is available through the existing `TorchOptimizer`.
- LBFGS was not included because the current `TorchOptimizer` wrapper does not implement the closure-style interface required by `torch.optim.LBFGS`.
- `label` initialization uses the reference `of_labels/spatial/coeffs[-1]` from the existing label as a warm-start, transformed into the model sample basis.

Main command:

```bash
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
/usr/bin/time -v -o _runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_sad_label_500_time.txt \
.venv/bin/python scripts/qm9_density_optimization_convergence_sweep.py \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --run EG_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt \
  --run EGF_lam1_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt \
  --output-json _runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_sad_label_500.json \
  --output-csv _runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_sad_label_500.csv \
  --summary-csv _runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_sad_label_500_summary.csv \
  --curves-csv _runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_sad_label_500_curves.csv \
  --sample-id 0 \
  --device cuda:0
```

Runtime:

- Exit status: 0
- Wall time: 10:31.13
- Max RSS: 1,644,028 KB

## Main Results

Convergence tiers:

- engineering: projected density-gradient norm `<1e-2`
- medium: `<1e-3`
- strict: `<1e-4`

At `max_cycle=500`, the best configs were:

| Run | Optimizer | lr | Init | Eng. converged | Mean final grad | Max final grad | Mean final energy |
|---|---|---:|---|---:|---:|---:|---:|
| EG_s3000 | Adam | 1e-3 | label | 2 / 5 | 0.031671 | 0.062851 | -240.631648 |
| EG_s3000 | SGD | 1e-3 | sad_default | 0 / 5 | 0.038355 | 0.048808 | -240.628900 |
| EG_s3000 | SGD | 1e-3 | label | 0 / 5 | 0.038615 | 0.049028 | -240.628818 |
| EG_s3000 | Adam | 1e-3 | sad_default | 2 / 5 | 0.047821 | 0.093619 | -240.630498 |
| EGF_lam1_s3000 | SGD | 1e-3 | sad_default | 0 / 5 | 0.045439 | 0.055842 | -241.049843 |
| EGF_lam1_s3000 | SGD | 1e-3 | label | 0 / 5 | 0.045760 | 0.056103 | -241.049765 |
| EGF_lam1_s3000 | Adam | 1e-3 | label | 0 / 5 | 0.072013 | 0.089895 | -241.052586 |
| EGF_lam1_s3000 | Adam | 1e-3 | sad_default | 0 / 5 | 0.107847 | 0.143979 | -241.049201 |

The only threshold crossings were in EG_s3000 with Adam:

| Run | Molecule | Optimizer | Init | Threshold | Cycle |
|---|---|---|---|---:|---:|
| EG_s3000 | 0000062 | Adam | label | 1e-2 | 293 |
| EG_s3000 | 0000062 | Adam | label | 1e-3 | 395 |
| EG_s3000 | 0000062 | Adam | label | 1e-4 | 488 |
| EG_s3000 | 0000062 | Adam | sad_default | 1e-2 | 325 |
| EG_s3000 | 0000062 | Adam | sad_default | 1e-3 | 450 |
| EG_s3000 | 0000171 | Adam | label | 1e-2 | 436 |
| EG_s3000 | 0000171 | Adam | sad_default | 1e-2 | 448 |

No EGF_lam1_s3000 molecule reached even the relaxed `<1e-2` threshold within 500 cycles.

## Max-Cycle Effect

For the most stable paired setting, SGD `lr=1e-3`, `sad_default`:

| Run | Max cycle | Eng. converged | Mean final grad | Max final grad | Mean final energy |
|---|---:|---:|---:|---:|---:|
| EG_s3000 | 50 | 0 / 5 | 0.612605 | 0.736105 | -240.428903 |
| EG_s3000 | 100 | 0 / 5 | 0.330686 | 0.401218 | -240.548573 |
| EG_s3000 | 200 | 0 / 5 | 0.150151 | 0.184233 | -240.606816 |
| EG_s3000 | 500 | 0 / 5 | 0.038355 | 0.048808 | -240.628900 |
| EGF_lam1_s3000 | 50 | 0 / 5 | 0.810039 | 0.915541 | -240.720387 |
| EGF_lam1_s3000 | 100 | 0 / 5 | 0.425853 | 0.482904 | -240.923154 |
| EGF_lam1_s3000 | 200 | 0 / 5 | 0.187487 | 0.211920 | -241.017138 |
| EGF_lam1_s3000 | 500 | 0 / 5 | 0.045439 | 0.055842 | -241.049843 |

This shows that `max_cycle=50` is clearly too short. However, simply raising to 500 is still not enough to reach `<1e-2` for all molecules.

## Learning Rate / Optimizer / Initialization

Findings:

- `lr=1e-3` is consistently best among the tested learning rates.
- `3e-4` and `1e-4` are too slow for 500 cycles; they often look plateau-like at 50-200 cycles and remain far above `<1e-2`.
- SGD `lr=1e-3` is the most stable paired choice for EG vs EGF. It gives monotonic/decreasing curves for all molecules and no failures, but no threshold crossings.
- Adam `lr=1e-3` helps EG on 2/5 molecules, but it is worse for EGF than SGD. It is not a good paired setting for EG vs EGF physical Hessian evaluation.
- `label` warm-start helps some EG Adam cases, but does not solve EGF convergence.
- No tested configuration diverged. At `max_cycle=500`, all 120 curves were classified as decreasing.
- At shorter budgets, many low-lr Adam curves were classified as plateau-like, which is consistent with the learning rate being too small for this objective scale.

## Displacement Check

Script added:

- `scripts/qm9_density_optimization_displacement_check.py`

After selecting the most stable paired setting, SGD `lr=1e-3`, `sad_default`, `max_cycle=500`, the same optimizer was tested on two molecules with `coord_idx=0`, `delta=1e-3 Bohr`, using base, plus, and minus geometries.

Command output:

- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/displacement_2mol_sgd1e3_sad500.json`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/displacement_2mol_sgd1e3_sad500.csv`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/displacement_2mol_sgd1e3_sad500_curves.csv`

Runtime:

- Exit status: 0
- Wall time: 1:41.25
- Max RSS: 1,519,360 KB

Results at threshold `<1e-2`:

| Run | Molecule | Side | Converged | Final grad | Final energy |
|---|---|---|---:|---:|---:|
| EG_s3000 | 0000010 | base | false | 0.033324 | -132.907231 |
| EG_s3000 | 0000010 | plus | false | 0.033168 | -132.902158 |
| EG_s3000 | 0000010 | minus | false | 0.033161 | -132.901671 |
| EG_s3000 | 0000062 | base | false | 0.029548 | -283.206399 |
| EG_s3000 | 0000062 | plus | false | 0.029552 | -283.206137 |
| EG_s3000 | 0000062 | minus | false | 0.029544 | -283.206671 |
| EGF_lam1_s3000 | 0000010 | base | false | 0.033357 | -133.021595 |
| EGF_lam1_s3000 | 0000010 | plus | false | 0.033257 | -132.995928 |
| EGF_lam1_s3000 | 0000010 | minus | false | 0.033244 | -132.991604 |
| EGF_lam1_s3000 | 0000062 | base | false | 0.038122 | -283.718140 |
| EGF_lam1_s3000 | 0000062 | plus | false | 0.038114 | -283.717880 |
| EGF_lam1_s3000 | 0000062 | minus | false | 0.038131 | -283.718407 |

The displaced structures behave almost identically to the base structures: stable, decreasing, but still not reaching `<1e-2`.

## 5000-Cycle Extension

The user-requested `max_cycle=5000` extension was run after the 500-cycle sweep. To keep the run focused, only the effective learning-rate region from the first sweep was retained:

- optimizer: SGD, Adam
- lr: `1e-3`
- initialization: `sad_default`, `label`
- models: EG_s3000, EGF_lam1_s3000
- molecules: same 5 `sample_id=0` original structures
- derived max-cycle budgets: 50, 100, 200, 500, 1000, 2000, 5000

Command output:

- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_lr1e3_sad_label_5000.json`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_lr1e3_sad_label_5000.csv`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_lr1e3_sad_label_5000_summary.csv`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_lr1e3_sad_label_5000_curves.csv`

Runtime:

- Exit status: 0
- Wall time: 21:16.27
- Max RSS: 1,662,376 KB

At `max_cycle=5000`, Adam converges all 5 original structures to strict `<1e-4` for both EG and EGF. SGD reaches engineering `<1e-2` for all 5 but still does not reach medium or strict.

| Run | Optimizer | Init | Eng. `<1e-2` | Medium `<1e-3` | Strict `<1e-4` | Mean final grad | Max final grad | Mean final energy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| EG_s3000 | Adam | sad_default | 5 / 5 | 5 / 5 | 5 / 5 | 8.689941e-05 | 9.704815e-05 | -240.632436 |
| EG_s3000 | Adam | label | 5 / 5 | 5 / 5 | 5 / 5 | 9.524287e-05 | 9.843930e-05 | -240.632419 |
| EG_s3000 | SGD | sad_default | 5 / 5 | 0 / 5 | 0 / 5 | 0.001542 | 0.002010 | -240.632291 |
| EG_s3000 | SGD | label | 5 / 5 | 0 / 5 | 0 / 5 | 0.001520 | 0.001992 | -240.632275 |
| EGF_lam1_s3000 | Adam | sad_default | 5 / 5 | 5 / 5 | 5 / 5 | 9.116099e-05 | 9.996814e-05 | -241.055078 |
| EGF_lam1_s3000 | Adam | label | 5 / 5 | 5 / 5 | 5 / 5 | 8.275258e-05 | 9.958839e-05 | -241.055016 |
| EGF_lam1_s3000 | SGD | sad_default | 5 / 5 | 0 / 5 | 0 / 5 | 0.002115 | 0.002803 | -241.054876 |
| EGF_lam1_s3000 | SGD | label | 5 / 5 | 0 / 5 | 0 / 5 | 0.002109 | 0.002804 | -241.054814 |

Budget progression for the practical non-label setting, Adam `lr=1e-3`, `sad_default`:

| Run | Max cycle | Eng. `<1e-2` | Medium `<1e-3` | Strict `<1e-4` | Mean final grad | Max final grad |
|---|---:|---:|---:|---:|---:|---:|
| EG_s3000 | 500 | 2 / 5 | 1 / 5 | 0 / 5 | 0.047821 | 0.093619 |
| EG_s3000 | 1000 | 5 / 5 | 2 / 5 | 2 / 5 | 0.001536 | 0.003151 |
| EG_s3000 | 2000 | 5 / 5 | 5 / 5 | 4 / 5 | 0.000248 | 0.000901 |
| EG_s3000 | 5000 | 5 / 5 | 5 / 5 | 5 / 5 | 8.689941e-05 | 9.704815e-05 |
| EGF_lam1_s3000 | 500 | 0 / 5 | 0 / 5 | 0 / 5 | 0.107847 | 0.143979 |
| EGF_lam1_s3000 | 1000 | 5 / 5 | 1 / 5 | 0 / 5 | 0.003782 | 0.005331 |
| EGF_lam1_s3000 | 2000 | 5 / 5 | 5 / 5 | 5 / 5 | 9.116099e-05 | 9.996814e-05 |
| EGF_lam1_s3000 | 5000 | 5 / 5 | 5 / 5 | 5 / 5 | 9.116099e-05 | 9.996814e-05 |

Strict convergence cycle range for Adam `lr=1e-3`, `sad_default`:

- EG_s3000: 617-2562 cycles across the 5 molecules.
- EGF_lam1_s3000: 1109-1931 cycles across the 5 molecules.

Interpretation:

- The previous failure was largely a `max_cycle` and optimizer-choice issue.
- Adam `lr=1e-3` is now the best tested optimizer setting.
- `sad_default` is preferred over `label` for subsequent displaced-geometry Hessian checks because it does not require a reference density for the displaced geometry.
- 5000 is a safe upper bound for this 5-molecule subset; in practice most strict convergence happens by 1000-2600 cycles.

## Adam 5000 Displacement Check

After Adam succeeded on the original structures, the 2-molecule displacement check was repeated with Adam `lr=1e-3`, `sad_default`, `max_cycle=5000`.

Command output:

- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/displacement_2mol_adam1e3_sad5000.json`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/displacement_2mol_adam1e3_sad5000.csv`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/displacement_2mol_adam1e3_sad5000_curves.csv`

Runtime:

- Exit status: 0
- Wall time: 3:12.12
- Max RSS: 1,556,464 KB

All 12 trajectories, `2 molecules x 2 models x base/plus/minus`, reached strict `<1e-4`.

| Run | Molecule | Side | Strict cycle | Final grad | Final energy |
|---|---|---|---:|---:|---:|
| EG_s3000 | 0000010 | base | 1240 | 9.931847e-05 | -132.909899 |
| EG_s3000 | 0000010 | plus | 1303 | 7.971781e-05 | -132.904804 |
| EG_s3000 | 0000010 | minus | 1296 | 9.958948e-05 | -132.904316 |
| EG_s3000 | 0000062 | base | 617 | 5.364750e-05 | -283.208573 |
| EG_s3000 | 0000062 | plus | 857 | 8.225731e-05 | -283.208312 |
| EG_s3000 | 0000062 | minus | 638 | 7.364665e-05 | -283.208845 |
| EGF_lam1_s3000 | 0000010 | base | 1447 | 9.049726e-05 | -133.024425 |
| EGF_lam1_s3000 | 0000010 | plus | 1619 | 9.053769e-05 | -132.998761 |
| EGF_lam1_s3000 | 0000010 | minus | 1483 | 8.576410e-05 | -132.994438 |
| EGF_lam1_s3000 | 0000062 | base | 1470 | 9.450223e-05 | -283.721148 |
| EGF_lam1_s3000 | 0000062 | plus | 1406 | 7.179029e-05 | -283.720887 |
| EGF_lam1_s3000 | 0000062 | minus | 1388 | 5.081165e-05 | -283.721415 |

## Answers

Can original structures converge under some settings?

- Yes. With Adam `lr=1e-3`, `max_cycle=5000`, both EG_s3000 and EGF_lam1_s3000 converge 5/5 original structures to strict `<1e-4`.
- With SGD `lr=1e-3`, both models reach engineering `<1e-2` by 5000 cycles, but neither reaches medium `<1e-3` or strict `<1e-4`.

Are EG and EGF convergence properties different?

- Yes at short budgets: EGF is slower than EG, especially at 500 cycles.
- With Adam `lr=1e-3` and enough cycles, both models reach strict convergence.
- Under SGD, EGF still has slightly higher final gradients than EG after 5000 cycles.

Is the current failure due to max_cycle, lr, optimizer, or plateau?

- `max_cycle=50` is definitely insufficient.
- `lr=1e-3` is the best tested value; lower lrs are too slow.
- Optimizer choice matters: Adam `lr=1e-3` solves strict convergence on this 5-molecule subset, while SGD `lr=1e-3` remains above strict threshold at 5000 cycles.
- At 500 cycles the best curves are still decreasing, not diverging or oscillating. The main issue was slow convergence / not enough effective optimization progress, not numerical blow-up.
- Some short-budget, low-lr Adam curves look plateau-like, but the best 500-cycle curves are decreasing.

Which setting is best for subsequent density-optimized Hessian eval?

- Adam `lr=1e-3`, `sad_default`, `max_cycle=5000`, strict threshold `1e-4`.
- This setting converges both models on the original 5 structures and on the 2-molecule `R +/- delta` check.
- `label` warm-start also works, but `sad_default` is cleaner for displaced geometries because it does not require a displaced reference density.

Can we continue physical Hessian, or only proxy Hessian?

- The density optimization convergence blocker is resolved for this 5-molecule subset.
- We can now rerun a small density-relaxed Hessian evaluation with converged density optimization using Adam `lr=1e-3`, `max_cycle=5000`.
- This still does not by itself prove a full total OFDFT physical Hessian, because the separate force-path audit still matters. It is the next step toward a converged density-relaxed Hessian check.

## Artifacts

- `scripts/qm9_density_optimization_convergence_sweep.py`
- `scripts/qm9_density_optimization_displacement_check.py`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_sad_label_500.json`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_sad_label_500.csv`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_sad_label_500_summary.csv`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_sad_label_500_curves.csv`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/displacement_2mol_sgd1e3_sad500.*`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/original_5mol_sgd_adam_lr1e3_sad_label_5000.*`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_density_optimization_convergence_sweep/displacement_2mol_adam1e3_sad5000.*`
