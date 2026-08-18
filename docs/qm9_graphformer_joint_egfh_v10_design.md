# QM9 Graphformer shared-optimizer joint E/G/F/H v10

Date: 2026-08-10

## Objective

Test the HORM-style optimization change without changing the physical model,
labels, derivative backend, probe distribution, or scalar-energy ownership.
Every update differentiates

`lambda_E L_E + lambda_G L_G + lambda_F L_F + lambda_H L_H`

once with respect to the Graphformer parameters and applies one shared AdamW
state. The frozen alternating v9 control instead uses eight H-only and two
E/G/F-only updates with two independent AdamW states.

## Why this is the next controlled experiment

The v7 clipping ablation showed that changing the clipped gradient magnitude
by two orders of magnitude did not materially change the AdamW trajectory. Its
handoff conclusion explicitly recommended either changing the H/replay
learning-rate ratio or using a common mixed E/G/F/H gradient with shared
optimizer state. The latter is also closest to HORM's combined objective and
preserves the calibrated H-to-EGF parameter-gradient ratio inside the actual
optimizer update.

No independent force or Hessian head is introduced. E/G remain evaluated at
the fixed label density; complete relaxed F/H remain evaluated at the strictly
converged model-self-consistent density and come from the same scalar energy.

## Training-length decision

Longer training is not justified unconditionally:

- Torch-autograd v9 improved Relative Frobenius from `1.9225741` to
  `1.8941366` in ten total steps (eight H updates), while total-energy absolute
  error regressed by `5.39%`.
- The 50-step cyclic-orthogonal v5 control improved Relative Frobenius only
  `6.91%`, from `1.9225732` to `1.7896523`, while energy error regressed
  `19.90%` and crossed the frozen 10% gate.
- A linear extrapolation of those short trajectories would require hundreds
  of updates to reach the `0.90` stage gate. That extrapolation is not a sound
  reason for an expensive open-ended run.

V10 therefore registers a matched ten-step formal comparison first. All ten
updates are joint and contain one fresh internal Rademacher HVP. A same-protocol
resume of 40 additional steps is registered but is conditional, not automatic.

Continue from step 10 to step 50 only when all of the following hold:

1. every density, electron-number, KKT response, finiteness, and symmetry gate
   passes;
2. energy and force error ratios remain at most `1.10` relative to step 0;
3. final Relative Frobenius is below the matched v9 endpoint `1.8941365720`;
4. no optimizer or derivative fallback is observed.

During the conditional extension, evaluate at steps 20, 30, 40, and 50. Stop
if Relative Frobenius fails to improve by at least `0.5%` over two consecutive
ten-step windows, or if either E/F regression ratio exceeds `1.10` at two
consecutive evaluations. Do not extend beyond 50 steps without a new protocol;
the prior evidence points to representation/capacity as the remaining limit.

## Executed result

The zero-learning-rate calibration and matched ten-step formal run completed
on node01 GPU 0. Calibration confirmed `update_kind=joint` and
`backpropagated_components=E;F;G;H`; it selected
`lambda_H=0.2647199267052535`. The formal run then performed ten joint updates
with one shared AdamW state and ten fresh internal Rademacher probes.

| metric | step 0 | joint step 10 | change | alternating v9 step 10 |
| --- | ---: | ---: | ---: | ---: |
| Relative Frobenius | 1.92257408 | 1.90471448 | -0.929% | 1.89413657 |
| Hessian MAE | 0.05695264 | 0.05629258 | -1.159% | 0.05610792 |
| Hessian RMSE | 0.13645060 | 0.13518306 | -0.929% | 0.13443231 |
| force MAE | 0.02881133 | 0.02823202 | -2.011% | 0.02866193 |
| energy absolute error | 0.00073360 | 0.00076494 | +4.272% | 0.00077316 |

Joint optimization improves E/F retention relative to alternating v9, but its
final Relative Frobenius is `0.01057791` higher (`0.558%` worse). On the same
raw-PBE vibrational evaluator, frequency MAE/RMSE are
`758.857/1156.758 cm^-1` for joint v10 versus
`757.482/1151.635 cm^-1` for alternating v9. Both retain three imaginary modes
versus one in PBE. Joint mean mode overlap is marginally higher
(`0.66204` versus `0.66096`), but this does not offset the worse curvature and
frequency errors.

The per-probe H loss is noisy rather than monotonically decreasing. At steps
1/5/10, the H-versus-aggregate-EGF gradient cosines are
`0.2041/0.0242/0.0633`: mostly weakly aligned, not strongly conflicting. The
shared optimizer is therefore dominated by the much larger force-gradient
history even though the initial H gradient was calibrated to 25% of aggregate
EGF. The experiment does not support the hypothesis that separate AdamW state
was the main Hessian bottleneck.

The registered extension criterion requiring a final Relative Frobenius below
the v9 endpoint is not met. The initial decision was therefore not to launch
the 40-step resume. The user explicitly authorized overriding that condition
on 2026-08-10, so a same-protocol continuation from cumulative step 10 to 50
was launched at
`/home/shenwei01/xzh_node02_20260724/runs/qm9_graphformer_egfh_torch_autograd_joint_egfh_v10/node01_joint_v10_resume_s10to50_20260810a`.
This continuation is an authorized diagnostic after a failed advancement
condition, not evidence that the condition passed. Final results are pending.

The user then authorized continuing to cumulative step 300. A detached
same-protocol orchestrator at
`scripts/monitor_and_extend_qm9_joint_v10_to_step300.sh` validates step 50,
chains 40-step resumes through step 290, and terminates the final registered
chunk only after the step-300 complete Hessian and restorable checkpoint have
both been written. State is recorded under
`step300_monitor_20260810a/status.json` in the v10 run family. A Codex thread
heartbeat checks this state every 30 minutes. No continuation beyond step 300
is authorized.

Calibration output:
`/home/shenwei01/xzh_node02_20260724/runs/qm9_graphformer_egfh_torch_autograd_joint_egfh_v10/node01_joint_v10_calibration_20260810a`.

Formal output:
`/home/shenwei01/xzh_node02_20260724/runs/qm9_graphformer_egfh_torch_autograd_joint_egfh_v10/node01_joint_v10_formal_s10_20260810a`.

## Changed files

- `scripts/qm9_complete_total_capacity_train.py`
- `scripts/launch_qm9_graphformer_implicit_relaxed_hvp_pilot_node02.sh`
- `configs/audit/qm9_graphformer_egfh_torch_autograd_joint_egfh_v10.yaml`
- `tests/test_qm9_complete_total_capacity_train.py`
- `scripts/monitor_and_extend_qm9_joint_v10_to_step300.sh`
