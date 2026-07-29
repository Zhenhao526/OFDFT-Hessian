# QM9 Frozen-Graphformer Structured Force Attention Pilot

Date: 2026-07-29 Asia/Shanghai

## Purpose

This is the preregistered force-first experiment proposed after the frozen
Graphformer Hessian attention head failed its transfer gate. It preserves the
requested model boundary:

```text
(nuclear coordinates R, complete density coefficients c)
    -> intact Graphformer backbone
    -> additional equivariant force attention readout
```

The source Graphformer, scalar energy head, and initial-density head are
unchanged. All 18,692,586 source parameters are frozen.

## Architecture

The new head reads the final 768-dimensional per-atom Graphformer states:

1. layer normalization;
2. `768 -> 3 x 128` query/key/value projection;
3. eight attention heads with 12-RBF distance bias;
4. attention context plus a projected residual atom state;
5. a 32-dimensional atom latent;
6. a symmetric pair MLP producing one scalar for each undirected atom pair;
7. equal-and-opposite scalar-times-unit-direction pair contributions.

The head has 443,593 trainable parameters, about 2.37% of the frozen backbone
parameter count. Its last layer is initialized to zero.

The construction gives exact atom-permutation covariance, rotation covariance,
translation invariance, zero molecular net force, and zero molecular net
torque. It is a direct force readout, not the derivative of a learned scalar
energy.

## Protocol

- source checkpoint:
  `/home/shenwei01/xzh_node02_20260724/models/train/runs/qm9_train800_egf_force1_s12330_rebuild_v1/checkpoints/last.ckpt`;
- source checkpoint SHA256:
  `ca45fda1eedb9815e9cac26c23f95f0a60666f73c89d4a0e967803ddc4690a45`;
- deterministic SHA256-ranked selection from the train800 split;
- 128 train-only parents: 96 fit and 32 held;
- four geometries per parent at the final SCF density;
- 384 fit samples and 128 held samples;
- 1,000 fixed training steps, batch size 16, AdamW, learning rate `1e-4`;
- no held-set checkpoint selection;
- no validation or Test100 access.

An initial 20-step implementation smoke test at `1e-3` showed unstable fit
optimization. The formal learning rate was reduced to `1e-4` using only fit
behavior. The subsequent smoke test decreased normalized fit loss from
`1.3467` to `0.7490`; no held result was used to tune the formal run.

## Formal result

| Split/model | Component MAE (Ha/Bohr) | Global relF |
|---|---:|---:|
| fit: new force head | 0.005046 | 0.84969 |
| fit: source energy derivative | 0.002375 | 0.42469 |
| fit: zero force | 0.006196 | 1.00000 |
| held: new force head | 0.005382 | 0.86199 |
| held: source energy derivative | 0.002353 | 0.38434 |
| held: zero force | 0.006486 | 1.00000 |

On held data, the new head improves component MAE over zero force by 17.02%,
but is 2.29 times worse than the existing energy-derived source force. All
performance gates fail. Exact constraints pass: maximum absolute net force is
`7.45e-9` and maximum absolute net torque is `2.98e-8`.

Parent-level paired results reinforce the aggregate result:

| Split | Parents | Head beats source | Head beats zero | Head median MAE | Source median MAE | Zero median MAE |
|---|---:|---:|---:|---:|---:|---:|
| fit | 96 | 0 | 96 | 0.005109 | 0.002314 | 0.006157 |
| held | 32 | 0 | 31 | 0.005321 | 0.002284 | 0.006206 |

The held maximum parent MAEs are `0.008372` for the head, `0.004029` for the
source, and `0.009668` for zero force.

Fit and held errors are close, while the fit error itself remains far above
the source result. The dominant failure is therefore representation/readout
underfitting rather than conventional train-to-held overfitting. The noisy
normalized minibatch loss (`1.395` at step 1, `0.510` at step 700, and `0.966`
at step 1,000) is consistent with this diagnosis.

## Decision

This checkpoint must not initialize the Hessian/response head. It does not
justify longer training, Graphformer unfreezing, or validation/Test100 access.

The experiment isolates a specific bottleneck: a central-pair equivariant
readout from the final invariant 768-dimensional atom states learns some force
signal, but discards too much derivative-relevant directional and
coefficient-level information to compete with differentiating the source
scalar energy.

The next small architecture experiment, if pursued, should retain the same
`(R,c) -> Graphformer` backbone while exposing intermediate G3D edge
messages/attention values or explicit vector channels to the force head.
For density response, it should tap the density branch before it is summed
with element and distance embeddings. A fit-only distillation check against
the source energy-derived forces can first determine whether such a readout
can reproduce information already present in the backbone before spending
clean PBE labels.

## Artifacts

Remote artifact:

`/home/shenwei01/xzh_node02_20260724/artifacts/graphformer_frozen_force_attention_pilot_v1_20260729`

Local copy:

`/Users/xia/Documents/Codex/2026-07-28/yue/outputs/graphformer_frozen_force_attention_pilot_v1_20260729`

SHA256:

- summary: `67fc46cc85338097a5da6e9a8ad1aa77694dbb711bdeb9d454e67ebca6ef4886`;
- per-sample CSV:
  `98ace22892582a8d40c6145f6d78347b30522aab3f704bbf25d57b3772f01449`;
- per-parent CSV:
  `a627eb7016a89e61a4df44b1117226277563432ff713553ae35023fae497d8d0`;
- registration:
  `e36eaa83592f72f7c5b3a3f5dd282c7fc4e1dba2d81b781e8d3c99408332a2c3`;
- checkpoint:
  `aef7e09a86005e53f07b2d6011fc566780835770235d72cb1f83041b4b2b0eec`.
