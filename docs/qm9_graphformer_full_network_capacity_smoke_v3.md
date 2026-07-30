# QM9 Graphformer Full-Network Hessian Capacity Smoke v3

Last updated: 2026-07-30 12:20 Asia/Shanghai

## Purpose

The earlier `energy_readout` capacity arm only optimized the final energy MLP
on top of a frozen Graphformer. It cannot establish whether the complete
Graphformer can express the target Hessian.

This v3 smoke therefore asks a narrower and correctly scoped question:

> Can the complete `R + c -> Graphformer -> scalar energy` model take a
> finite Hessian-only optimization step on all 39 internal directions of one
> training molecule?

A relative Frobenius error at or below `0.05` is a positive single-molecule
capacity result. Failure, no improvement, or not reaching the threshold in one
update is not evidence that Graphformer lacks Hessian capacity.

## Frozen experiment

- Molecule/sample: `0028399/0`, training data only.
- Coordinate space: complete 39-dimensional internal subspace with translation
  and rotation removed.
- Trainable scope: all 79 Graphformer parameter tensors, 18,692,586 parameters.
- Scalar owner: the complete OFDFT energy; no force or Hessian output head.
- Derivative: analytic geometry response plus constrained density/KKT response
  and implicit parameter adjoint.
- Loss: full internal-Hessian relative Frobenius squared. Energy, force, and
  density loss weights are zero.
- Optimizer budget: one AdamW update, learning rate `1e-5`, gradient clipping
  at `10`.
- Initialization arms:
  - `pretrained`: the registered force-trained checkpoint.
  - `scratch`: deterministic reset with seed `20260730`; the neural modules are
    reset while atom references and dimension scaling are preserved.
- Data boundary: no stable5, train20, held directions, validation, or test100
  access.

Protocol:
`configs/audit/qm9_graphformer_0028399_full_network_capacity_smoke_v3.yaml`

## Preflight and one-direction gradient results

Both initialization arms passed strict density convergence and the no-update
one-direction full-network gradient smoke.

| Initialization | Density residual | Gradient norm | Parameter tensors with gradient | Parameter change | Implicit-adjoint relative residual | Peak GPU memory | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| pretrained | `1.19e-10` | `24.3189` | 74 / 79 | `0` | `5.24e-15` | `68001.8 MB` | `242.2 s` |
| scratch | `6.62e-11` | `20.6564` | 74 / 79 | `0` | `2.93e-15` | `68001.8 MB` | `269.7 s` |

The scratch parameter-state hash differs from the pretrained state and is
deterministically reproducible:

- pretrained: `7bd51dbef9f5ac3dbf8a4c503f25c1197597e508033fc34436c0ea7dad6170e6`
- scratch: `1c08708148f81f87e195cce48419cd979c55e99c7815625605f471a98f59916b`

These results establish that the high-order training graph reaches the full
Graphformer and is numerically executable for both initializations. They do
not yet establish Hessian capacity.

## Formal one-update status

The sequential pretrained/scratch full39 run was launched on the otherwise
idle GPU 7 on node02. The supervisor is:

`scripts/launch_qm9_graphformer_full_network_capacity_smoke_node02.sh`

Run root:

`/home/shenwei01/xzh_node02_20260724/runs/graphformer_0028399_full_network_capacity_smoke_v3`

The launcher fails closed per arm and writes `smoke_summary.json` after both
arms finish. Given the measured one-direction cost, the full39 parameter
gradient is an hours-scale calculation even for one update.

## Interpretation boundary

The experiment can support either of these statements:

- If an arm reaches the gate: the complete Graphformer has demonstrated
  single-molecule Hessian capacity under that initialization.
- Otherwise: the one-update optimizer budget did not demonstrate the capacity
  gate.

It cannot support the statement “Graphformer cannot express the target
Hessian.” A negative architecture conclusion would require a real
optimization/capacity curve with multiple budgets and learning rates, not a
single final-layer fit or a single full-network update.
