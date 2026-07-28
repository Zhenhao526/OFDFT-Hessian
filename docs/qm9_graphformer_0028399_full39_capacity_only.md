# Graphformer 0028399 Full39 Hessian Capacity-Only Experiment

Last updated: 2026-07-27 22:08 Asia/Singapore

## Purpose

This is a new, single-train-parent capacity branch. It asks only whether the fresh Graphformer
parameterization can fit the complete internal PBE Hessian of `0028399`. It does not test
energy/force retention, direction generalization or parent generalization.

The previous hybrid relaxed-HVP performance failure is preserved and is not overwritten:

```text
old formal audit:
/home/shenwei01/xzh_node02_20260724/runs/graphformer_hybrid_relaxed_hvp_rebuild_v1/audit_0028399_formal_eigh_physical_classical_v1
old conclusion: correctness passed; performance gates failed
old full39 relative Frobenius: 2.57598
```

## Frozen Boundary

```text
protocol:
configs/audit/qm9_graphformer_0028399_full39_capacity_only_v2.yaml
protocol SHA256:
7359095fc16e5378f1c7700747ac50ba1dbd84ce7f4e89e1c182123edb26f785

source checkpoint:
/home/shenwei01/xzh_node02_20260724/models/train/runs/qm9_train800_egf_force1_s12330_rebuild_v1/checkpoints/last.ckpt
source checkpoint SHA256:
ca45fda1eedb9815e9cac26c23f95f0a60666f73c89d4a0e967803ddc4690a45

molecule/sample: 0028399 / 0
direction count: 39
direction role: capacity_train for all 39
stable5 access: false
train20 access: false
held-direction access: false
validation access: false
Test100 access: false
```

The one-parent assets were built directly from the frozen train-only split, the
`0028399.0000000` label and the explicit PBE Hessian file. The asset builder does not read a
stable5 or train20 manifest.

```text
asset registration SHA256:
3286b301fd21d520e6db530b486bf7229a2732883b054768c6ab2269a652c1a0
parent manifest SHA256:
a4f8c1b105ed5197284b5a7379bce15b351c40cc6f442ccbb737791935fa748f
direction manifest SHA256:
de658a63061be382b6ec5aa33e09e91e8abe3b86b079c9766c73117792a82aee
direction artifact SHA256:
266d510a3f581a06e593093557c9c330990d1906070dcb63cf0a82282317fff0
density-solver rescue SHA256:
a96f00e1a085e7fd304210792d69c0017198551485f22acef53414e8f6ca22ec
```

## Derivative Definition

The experiment uses the audited hybrid analytic complete-total relaxed-HVP:

```text
G_y y_v = -G_R v
H_relaxed v = L_RR v + L_Ry y_v
```

- the center density must have projected gradient below `1e-8`;
- density/KKT tangent and parameter adjoint use dense direct solves;
- classical energy uses the physical basis;
- natural reparametrization uses explicit `eigh_second_order_audit`;
- fixed-density, stale-density and detached-HVP fallbacks are forbidden;
- force and HVP remain derivatives of the same complete scalar total energy.

This remains a hybrid method because PySCF first-integral derivatives are differentiated with
the registered directional center difference. It is not a fully analytic libcint
second-integral implementation.

## Capacity Objective

Only the complete internal Hessian objective is active:

```text
lambda_E = 0
lambda_F = 0
lambda_rho = 0
lambda_H = 1
```

For one parameter update, all 39 orthonormal internal directions are evaluated in order. Their
parameter VJPs are accumulated while parameters remain fixed, and the optimizer is allowed to
update only after direction 39. Therefore one update represents the exact complete internal
Hessian Frobenius gradient, not a stochastic direction gradient.

The pass criterion is only:

```text
full39 relative Frobenius <= 0.05
```

On the first pass, the checkpoint and summary are frozen and all remaining arms stop. No
energy/force joint training is authorized by this branch.

## Arms

Each optimizer arm starts independently from the registered fresh checkpoint:

1. energy readout;
2. readout plus final Graphformer block;
3. readout plus final two Graphformer blocks;
4. full Graphformer.

Each scope compares:

- AdamW;
- full-batch L-BFGS with strong-Wolfe line search;
- damped Gauss-Newton/Levenberg-Marquardt.

A matrix-free damped CGLS audit estimates the parameter-to-full39 residual Jacobian range once
per fresh scope. A nonconverged solve is reported only as an achieved linearized residual upper
bound; it is not called the minimum. A claim of insufficient capacity remains forbidden unless
full Graphformer, all optimizers and a converged linearized minimum all remain above 5%.

## Implementation

```text
scripts/prepare_qm9_graphformer_0028399_capacity_assets.py
scripts/qm9_graphformer_full39_capacity_only.py
scripts/launch_qm9_graphformer_0028399_capacity_only_node02.sh
configs/audit/qm9_graphformer_0028399_full39_capacity_only_v1.yaml
configs/audit/qm9_graphformer_0028399_full39_capacity_only_v2.yaml
configs/audit/qm9_graphformer_0028399_density_solver_rescue_v1.yaml
tests/test_qm9_graphformer_full39_capacity_only.py
tests/test_prepare_qm9_graphformer_0028399_capacity_assets.py
```

The shared analytic path now passes only `requires_grad=True` Graphformer parameters into the
implicit density parameter-adjoint function. This is required for readout/block freezing and
does not change the full-Graphformer first-order path.

The runner enforces process-wide `torch.float64` before constructing samples
or integral tensors. Every formal summary/checkpoint records implementation
and registered-asset SHA256 provenance. Failed strict-density attempts now
retain their final coefficients, full optimization curve and metadata.

Node02 focused tests pass `17/17` after the numeric/device fixes.

## Runtime Status

All failed preflights remain preserved. The apparent strict-density blocker
was an implementation omission: the old audit set the process-wide default
dtype before sample construction, while the initial capacity runner converted
only model parameters to float64. The resulting mixed numeric path imposed a
projected-gradient floor near `1e-5`:

```text
Adam endpoint:                         1.910055e-4
Adam -> L-BFGS endpoint:               1.445948e-2
Adam -> L-BFGS -> Newton(20):          1.413149e-5
Adam -> Newton(20), no L-BFGS:         1.124130e-5
```

After enforcing the registered process-wide float64 mode, the same checkpoint
and center geometry reached `9.871635e-11` in 306 cycles and 19.27 seconds.
The one-direction, no-update readout-adjoint smoke also passed:

```text
gradient norm: 3.252505
parameter change norm: 0
density residual: 1.183696e-10
KKT stationarity residual: 9.261774e-13
implicit-adjoint relative residual: 6.000259e-15
wall time: 92.01 s
peak GPU allocation: 62051 MiB
```

The first fresh full39 attempt completed the expensive direction graph and
then failed closed at the new runner's final projection assembly. The audited
relaxed-HVP is returned on CPU, while the newly created internal projector was
on CUDA. No physical result was emitted. The failure is retained at
`fresh_full39_float64_v1`.

Projection tensors now follow the HVP device and dtype. A second no-update
smoke exercised the exact formal projection and parameter-adjoint path:

```text
HVP device: cpu
formal projection path checked: true
gradient norm: 9.359230
parameter change norm: 0
density residual: 1.185125e-10
KKT stationarity residual: 1.235483e-12
implicit-adjoint relative residual: 5.379977e-15
```

The corrected fresh-checkpoint full39 evaluation completed:

```text
full39 relative Frobenius: 2.5759803326
full39 MAE: 0.0745226257
full39 RMSE: 0.2263722007
symmetry max abs: 1.207819e-5
antisymmetric/symmetric Frobenius: 1.229570e-5
density residual: 1.192500e-10
maximum response residual: 2.167498e-12
full39 wall time: 1119.53 s
total process wall time: 1136.38 s
max RSS: 70301.60 MiB
summary SHA256:
0e3b775ad3fa6f973336adc4afda4ef0c3eda2ce07c4a732cf28e7f8ef89e18f
array SHA256:
417b2d78185ceed91a5fa8d2111eb139139d9661f823cf3e98739173ecb717a8
```

This reproduces the untouched old baseline and is above the 5% gate. The
sequential capacity supervisor is now active on node02:

```text
supervisor PID: 3028160
active arm: energy_readout + AdamW
active child PID: 3028168
run:
/home/shenwei01/xzh_node02_20260724/runs/graphformer_0028399_full39_capacity_only_v2/energy_readout/adamw
```

The active arm starts independently from checkpoint SHA256 `ca45...0a45`.
Its complete update-0 evaluation is now durable:

```text
update 0 relative Frobenius: 2.5759803325
update 0 checkpoint SHA256:
ca8ef66b769f3efe1e8a4c7dbbb867c983f990e5575b1a64792a0c54f0a82120
```

The same arm remained healthy through update 18 on 2026-07-28 15:55
Asia/Singapore. It completed one exact 39-direction gradient accumulation and
one full39 evaluation per update:

```text
                           update 0       best/update 17   latest/update 18
full39 relative Frobenius  2.5759803325   0.8665426540     0.8735228965
full39 MAE                 0.0745226257   0.0324138950     0.0328187421
full39 RMSE                0.2263722007   0.0761501030     0.0767635133
gradient norm              n/a            13.727111        10.293410
update norm                n/a            0.022799532      0.021125553
density residual           1.09e-10       1.18e-10         1.20e-10
maximum response residual  1.84e-12       1.46e-12         1.82e-12
symmetry max abs           1.21e-5        2.05e-5          2.11e-5
```

The latest result is a 66.09% reduction from the fresh relative Frobenius
baseline, while the best evaluated result is a 66.36% reduction. Update 18 is
0.81% worse than update 17, so this is a local oscillation rather than a new
best checkpoint. The best error remains 17.33 times above the 0.05 capacity
gate. No capacity freeze marker exists and no capacity conclusion is
authorized.

The durable update-18 checkpoint SHA256 is
`5589cf31d43fdc9e8919c6acca0b7e0fa848a9097809b9439643378e9d14fdd0`.
At 15:57, update 19 was in the gradient phase at direction 2/39. The process
was active on node02 GPU 0 with about 53 GiB GPU memory and 76 GiB host RSS.
Stable5, train20, held directions, validation and Test100 remain locked.

## Commands

Asset registration:

```bash
python scripts/prepare_qm9_graphformer_0028399_capacity_assets.py \
  --protocol configs/audit/qm9_graphformer_0028399_full39_capacity_only_v2.yaml \
  --output-dir \
  /home/shenwei01/xzh_node02_20260724/artifacts/graphformer_0028399_full39_capacity_only_v2
```

After the full39 baseline preflight passes, the frozen sequential sweep is:

```bash
CAPACITY_DEVICE=cuda:0 \
bash scripts/launch_qm9_graphformer_0028399_capacity_only_node02.sh
```

The supervisor runs AdamW, L-BFGS and LM independently from the fresh
checkpoint for each scope, then the fresh-checkpoint Jacobian audit. It stops
immediately when any formal arm reaches 5%.
