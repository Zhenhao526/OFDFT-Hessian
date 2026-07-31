# EGFH P0 density-response audit

Date: 2026-07-31

Scope: one train-only parent, one internal direction, three checkpoints

Status: complete; no training, validation access, or Test100 access

## Decision

Do not scale or extend the present ten-parent fixed-endpoint EGFH objective.
The step-100 and step-1300 checkpoints do not learn a stable
density-relaxed Hessian. Their fixed-density curvature and density-response
correction become individually much larger than the final relaxed HVP and
nearly cancel. Step 1300 also loses most of the directional improvement seen
at step 100.

This is a rejection of the current Hessian loss/branch alignment, not of the
scalar `(R,c) -> Graphformer -> E_total` architecture. The next Hessian
experiment should supervise randomized internal-space implicit relaxed HVPs
on the same self-consistent model-density branch used at inference.

## Question and frozen scope

The audit tests whether the released article checkpoint and the current EGFH
step-100 and step-1300 checkpoints have a numerically stable local
Schur-complement decomposition

```text
H_relaxed v = E_RR v - E_Rc E_cc^-1 E_cR v
            = partial HVP + density-response correction.
```

It uses:

- train parent `0016298`;
- internal direction 0 from the frozen train20 direction asset;
- released article checkpoint, EGFH step 100, and EGFH step 1300;
- projected density-gradient threshold `1e-8`;
- centered force differences at `h = 1e-3, 3e-4, 1e-4 Bohr`;
- dense projected `E_cc` eigenspectrum and constrained direct response solve;
- float64 and `eigh_second_order_audit`.

The parent and direction manifests, molecule, direction, numerical settings,
checkpoint paths, and checkpoint hashes are frozen in
`configs/audit/qm9_egfh_p0_density_response_audit_v1.yaml`. No parameter
update is possible in the audit runner.

## Main results

The PBE reference HVP norm for this direction is `0.515174 Ha/Bohr^2`.

| checkpoint | center cycles | min eig(`E_cc`) | cond(`E_cc`) | `||E_RR v||` | `||response||` | `||H_relaxed v||` | response / relaxed | relaxed vs PBE rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| article | 660 | 2.047e-4 | 4.884e6 | 1.6390 | 0.8471 | 1.2648 | 0.670 | 2.3130 |
| step 100 | 10,475 | 1.303e-4 | 3.052e6 | 58.0518 | 57.7933 | 0.7905 | 73.106 | **1.3052** |
| step 1300 | 10,273 | 1.186e-4 | 3.780e6 | 13.3439 | 13.4743 | 1.0497 | 12.837 | 2.0640 |

All three projected density Hessians are positive definite under the frozen
relative-zero rule, but all are highly conditioned. The minimum eigenvalue
falls from `2.047e-4` in the article model to `1.186e-4` at step 1300.
Condition number alone does not explain the checkpoint trend because the
maximum eigenvalue also changes substantially.

The more decisive signal is cancellation:

| checkpoint | partial vs PBE rel-L2 | partial/response cosine | cancellation index `(||partial||+||response||)/||relaxed||` | density response/secant norm at `h=1e-4` |
| --- | ---: | ---: | ---: | ---: |
| article | 2.5999 | -0.6497 | 1.97 | 5.3683 |
| step 100 | 111.9811 | -0.999917 | 146.54 | 9.1776 |
| step 1300 | 25.1929 | -0.996983 | 25.55 | 25.6616 |

Step 100 improves the final directional error by `43.6%` relative to the
article checkpoint, but does so through near-exact cancellation of two
roughly `58`-norm terms. Step 1300 is `58.1%` worse than step 100 on the same
direction and retains near-antiparallel large-term cancellation. It is only
`10.8%` better than the article checkpoint on final directional error.

The learned density response itself grows monotonically on this direction:
the `h=1e-4` secant norm is `5.37 -> 9.18 -> 25.66`. This trend, together with
the falling smallest density-curvature eigenvalue, shows that prolonged EGFH
training is reshaping the density-response branch even when the final
relaxed-HVP error does not improve.

## Numerical correctness

The result is not a loose-density or finite-difference artifact.

| checkpoint | response stationarity residual | relaxed analytic vs strict FD at `h=1e-3` | at `h=3e-4` | at `h=1e-4` |
| --- | ---: | ---: | ---: | ---: |
| article | 4.42e-13 | 4.19e-5 | 7.37e-6 | 4.45e-7 |
| step 100 | 4.17e-13 | 1.37e-4 | 2.74e-5 | 1.52e-6 |
| step 1300 | 1.38e-12 | 4.94e-5 | 1.03e-5 | 3.98e-7 |

All 18 displaced endpoints pass the `1e-8` strict density threshold.
At `h=1e-4`, the analytic response and reoptimized force difference agree
between `4e-7` and `1.6e-6` relative L2. The direct constrained response
residuals are below `1.4e-12`.

Optimization cost is nevertheless checkpoint-sensitive. At `h=1e-3`, the
article endpoints require `274/279` cycles, step 100 requires
`10115/10112`, and step 1300 requires `215/219`. The smaller response-seeded
steps converge in 8--222 cycles. Step 100 therefore creates a particularly
unfriendly finite neighborhood despite its lower final directional HVP
error.

## Interpretation

The current training Hessian term uses conservative force differences at
displaced KS-density configurations. Physical evaluation instead requires
derivatives after minimizing the learned scalar functional with respect to
its own density coefficients. These are different branches.

The P0 result demonstrates the practical failure mode:

1. the force-secant objective can reduce a final directional error;
2. it does not separately control fixed-density curvature, mixed
   geometry-density derivatives, or the learned density response;
3. those terms can grow by one to two orders of magnitude and cancel;
4. continued training can lower the training objective without preserving
   the best relaxed-HVP checkpoint.

This also explains why the step-1300 full Hessian and vibrational metrics
remained poor even after the training loss appeared converged.

## Next lowest-cost experiment

Keep E/G/F and the scalar Graphformer energy owner, but replace the current H
term in a small train-only pilot:

1. sample one fresh Rademacher direction in the `3N-6` internal basis;
2. strictly relax the model center density or use a certified cached center;
3. solve the constrained density response for that direction;
4. train directly on the implicit relaxed-HVP target;
5. log `E_cc` minimum eigenvalue/condition proxy, response residual,
   response norm, correction/relaxed ratio, and partial-response cosine;
6. reject updates/checkpoints that develop negative density-curvature modes,
   failed response solves, or extreme cancellation;
7. select checkpoints by a frozen multi-direction train-only relaxed-HVP
   screen, not by total training loss alone.

Before training, an analytic-only extension to three train parents and three
directions each can determine robust cancellation thresholds. The expensive
three-displacement verification need not be repeated on every point because
the present P0 has already verified the analytic implementation; one strict
FD sentinel per checkpoint is sufficient.

An auxiliary density-response head remains feasible as an initializer or
preconditioner for `dc*/dR v`, but it must not own the final force or Hessian.

## Cost and provenance

- Total wall time: `1304.97 s` (`21.75 min`)
- Peak GPU memory: `834.74 MiB`
- GPU: one device on node02
- Formal result:
  `/home/shenwei01/xzh_node02_20260724/runs/qm9_egfh_p0_density_response_audit_v1/0016298_d0_preflight_v2_20260731`
- Formal summary SHA256:
  `6f343b83ecd789468271c91539aca7e8df6dc6bfbe4cb6dee2a95a9985bac241`
- Article/step100/step1300 summary SHA256:
  `05a9e2b3d783c5df19f9487d6631a8353e7bc52f2ec9d3053ce17181b34247cc`,
  `21741946d09909a033cf990a65874fc7a564de097e51caf573d336a32b21986e`,
  `5944ecfe3732b9373b5bcbd0c421ade88bbec29385be36d6864c5a505a650d23`
- Article/step100/step1300 array SHA256:
  `97d9043a66d2247dbabaae540c2fcb7e14d19f2d113bd28f3d9382e55778ef24`,
  `e1a5c5d84cd14a7a354263d8ff9eeeaa36f21012f60aba79a44372dceea3cb97`,
  `6fc246aaf77c6cb4793036f8e2bab83a71c607f46e41b4cbf0f27de424acd508`
- Frozen protocol SHA256:
  `1cd14dd7ba39fcf701f26a52eb9b7a0728636080f382f1c75c5ebcd5587fea36`
- Executed runner SHA256:
  `e29be896821aeebce0414ab8274b4be04edba7cd138d2eda839571c850868bab`
- Launcher SHA256:
  `c335d47a5ac9ea7b3852bec09a66464e4b97348130ff19851bca398e9ee76a80`

The first preflight stopped before scientific computation because an adapted
article checkpoint lacked required model keys and the requested
second-order matrix-power mode name was invalid. The formal rerun uses the
raw released checkpoint, validates all hashes, has no failures, and is the
only run used above. The checked-in runner adds fail-close checks for
non-finite tensors and response residuals plus progress logging after the
formal run; these do not change the numerical path, and all formal results
pass the added gates.
