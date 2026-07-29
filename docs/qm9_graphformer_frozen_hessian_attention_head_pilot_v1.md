# QM9 Frozen-Graphformer Structured Hessian Attention Pilot

Date: 2026-07-29 Asia/Shanghai

## Purpose

This pilot implements the requested architecture boundary:

```text
(nuclear coordinates R, complete density coefficients c)
    -> intact Graphformer backbone
    -> additional structured Hessian attention readout
```

It does not replace Graphformer with an independent geometry network. The
registered train800 checkpoint, energy head, initial-density head, and all
18,692,586 backbone parameters remain unchanged.

## Architecture

The source Graphformer contains:

- atom-hot embedding of the complete density coefficients;
- element and distance embeddings;
- 768-dimensional atom states;
- four G3D layers with 32 attention heads per layer;
- the original scalar energy and initial-guess readouts.

The new readout consumes the final 768-dimensional atom states:

1. layer normalization;
2. `768 -> 3 x 128` query/key/value projection;
3. eight attention heads with 12-RBF distance bias;
4. attention context plus a projected residual atom state;
5. a 16-dimensional context-aware atom latent;
6. the covariant `3 x 3` pair/node block Hessian readout;
7. exact symmetrization and molecular-internal projection.

The new head has 446,403 parameters, about 2.39% of the backbone parameter
count. Its final block coefficients are zero-initialized, so attaching it does
not change the source energy prediction.

## Frozen boundaries

- source checkpoint:
  `/home/shenwei01/xzh_node02_20260724/models/train/runs/qm9_train800_egf_force1_s12330_rebuild_v1/checkpoints/last.ckpt`;
- source SHA256:
  `ca45fda1eedb9815e9cac26c23f95f0a60666f73c89d4a0e967803ddc4690a45`;
- Graphformer trainable parameter count: exactly zero;
- train20 sample-0 parents only;
- fold 0 only for the small-scope result;
- no validation or Test100 access;
- no held-parent checkpoint selection.

Feature extraction recovered all 20 parents. Final atom states have dimension
768, and the maximum batch-versus-Hessian-asset geometry difference is
`4.67e-7 Bohr`.

## Tests

Eight implementation tests pass:

- zero initialization gives an exactly zero Hessian;
- rotation covariance;
- atom-permutation covariance;
- exact Hessian symmetry;
- translational and rotational null modes;
- regression tests for the underlying structured block readout.

## Fold-0 results

| Model | Steps | Fit median relF | Held median relF | Held P90 relF | Held max relF |
|---|---:|---:|---:|---:|---:|
| Geometry-only structured baseline | 800 | 0.221091 | 0.254864 | 0.274051 | 0.278928 |
| Frozen Graphformer attention | 200 | 0.229751 | 0.253175 | 0.265617 | 0.270501 |
| Frozen Graphformer attention | 800 | 0.174962 | 0.250678 | 0.292512 | 0.303243 |

At 800 steps the attention head improves fit capacity substantially, but the
held median improves by only 1.64%, below the preregistered 5% gate. The held
tail becomes worse.

Paired held-parent results:

| Parent | Geometry-only | Attention s800 | Change |
|---|---:|---:|---:|
| `0049017` | 0.247058 | 0.303243 | +0.056185 |
| `0082761` | 0.262670 | 0.267473 | +0.004802 |
| `0121249` | 0.278928 | 0.233884 | -0.045045 |
| `0124009` | 0.183464 | 0.218755 | +0.035291 |

The attention head wins one case and loses three. Mean relative-Frobenius
error increases by `0.01281` across the four held parents. Matrix symmetry is
exact and maximum external-mode leakage remains below `6.7e-16`.

## Decision

The final Graphformer states contain extra Hessian fitting capacity, but the
current attention readout does not convert it into stable cross-parent
improvement. The result does not justify:

- running the remaining four folds;
- unfreezing the final G3D layer;
- accessing validation or Test100.

The failure is not caused by loss of the requested `(R,c) -> Graphformer`
backbone. It occurs after that backbone, in the transferability of the
derivative readout.

Before any backbone unfreezing, the next useful small experiment should train
an equivariant force-attention head on the much larger clean train800 force
set, then initialize the Hessian/response head from that derivative-aware
representation. A response-specific variant should expose the density branch
before it is summed with element and distance embeddings, or retain
coefficient-level response tokens, rather than relying only on the final atom
state.

## Artifacts

Remote 800-step artifact:

`/home/shenwei01/xzh_node02_20260724/artifacts/graphformer_frozen_hessian_attention_head_fold0_s800_v1_20260729`

Summary SHA256:

`873f5e00a2cf4187aa225139fa058eb485f90aa235612b526c2e3b9ee25982c8`

Checkpoint SHA256:

`dda188ea00f16178f13faa5ef051d9c20b8d997af1605c8682aad35fc7370ba5`
