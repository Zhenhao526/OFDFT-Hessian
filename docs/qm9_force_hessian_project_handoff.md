# QM9 Force/Hessian Project Handoff

Last updated: 2026-07-30 10:51 Asia/Singapore

## Purpose

This is the handoff and continuity document for the QM9 P1-410 and random1000 force/Hessian work. Update this file after every major code change, experiment milestone, evaluator change, model checkpoint, or conclusion change.

Detailed reports remain in separate files under `docs/`; this document is the first file a new maintainer should read.

## Current Scope

Active scope is still the P1-410 early pilot plus a remote random1000 8-GPU training validation:

- dataset: 410 QM9 molecules, 1640 labels;
- models: EG baseline and EGF lambda=1.0 are the main comparison;
- promotable force must be derived from scalar model energy:
  `F_pred = -dE_pred/dR`;
- one independent structured force-head diagnostic has now been run, but it
  failed the source-force transfer gate and is not a promotable model;
- random1000 labels have been generated and used for a 10-epoch EG/EGF training validation;
- do not delete existing labels, checkpoints, cached labels, or Hessian references.

Current Hessian conclusions do not yet generalize to final P1/P2.

Full-scale training code preparation has started, but the full 133,885-molecule QM9 force-label
dataset has not been generated and no full-QM9 EG/EGF training job has been launched.

### Graphformer-preserving derivative-head pilots (2026-07-29)

Three train-only pilots now delimit what can and cannot be learned by adding
structured derivative readouts while keeping the main OFDFT work isolated from
validation and Test100.

#### Structured Hessian/density head

The first pilot is an independent, small structured Hessian control rather than
a replacement mainline model. It assembles equivariant `3 x 3` Cartesian
blocks, enforces exact symmetry, and projects out rigid translations and
rotations. Five-fold parent-isolated evaluation uses the 20 clean train
parents, sample 0 only.

| Variant | Parameters | fit median relF | held median relF | held P90 | held max |
| --- | ---: | ---: | ---: | ---: | ---: |
| geometry only | 11,979 | 0.225775 | 0.249269 | 0.283948 | 0.324290 |
| geometry + density summary | 13,515 | 0.225658 | 0.247198 | 0.288309 | 0.319053 |

The six-scalar-per-atom density summary improves held median by only `0.8309%`,
below the frozen `5%` gate, with 11 parent wins and 9 losses. Both variants
beat a zero Hessian on all 20 held parents and satisfy the exact structural
constraints, so structured direct Hessian output is feasible, but the current
density-value summary does not establish useful density-response information.
It is neither a Schur-complement response head nor a conservative
total-energy Hessian.

Artifact:
`/home/shenwei01/xzh_node02_20260724/artifacts/structured_density_hessian_head_pilot_v1_20260729`.
Formal summary SHA256:
`f4769f2ac5a90016be023e7e4e3a818f1a4e6ab47fef5111f0fec060afec1f73`.
Detailed report:
`docs/qm9_structured_density_hessian_head_pilot_v1.md`.

#### Frozen Graphformer Hessian attention head

The corrected architecture preserves the complete `(R,c) -> Graphformer`
backbone and adds an eight-head distance-biased structured Hessian readout from
the final 768-dimensional atom states. All 18,692,586 backbone parameters,
the scalar energy head, and the initial-density head remain unchanged and
frozen; the new head has 446,403 parameters.

On fold 0, 800 training steps reduce fit median relF to `0.174962`, but held
median is `0.250678` versus `0.254864` for the matched geometry-only control:
only a `1.64%` improvement. It wins 1 of 4 held parents, while held P90 and
maximum error worsen to `0.292512/0.303243`. The remaining four folds and
Graphformer unfreezing are therefore not authorized.

Artifact:
`/home/shenwei01/xzh_node02_20260724/artifacts/graphformer_frozen_hessian_attention_head_fold0_s800_v1_20260729`.
Formal summary SHA256:
`873f5e00a2cf4187aa225139fa058eb485f90aa235612b526c2e3b9ee25982c8`.
Detailed report:
`docs/qm9_graphformer_frozen_hessian_attention_head_pilot_v1.md`.

#### Frozen Graphformer force attention head

The force-first pilot keeps the same frozen Graphformer but adds a 443,593
parameter equivariant attention head. It forms symmetric atom-pair scalars and
equal-and-opposite scalar-times-unit-direction contributions, giving exact
permutation/rotation covariance, translation invariance, zero net force, and
zero net torque. Focused implementation tests pass `5/5`.

The formal train800-only experiment deterministically selects 128 parents:
96 fit and 32 held, with four geometries per parent (512 samples total). It
uses 1,000 fixed steps without held checkpoint selection and reads neither
validation nor Test100.

| Split/model | component MAE (Ha/Bohr) | global relF |
| --- | ---: | ---: |
| fit: new force head | 0.005046 | 0.84969 |
| fit: source energy derivative | 0.002375 | 0.42469 |
| fit: zero force | 0.006196 | 1.00000 |
| held: new force head | 0.005382 | 0.86199 |
| held: source energy derivative | 0.002353 | 0.38434 |
| held: zero force | 0.006486 | 1.00000 |

The head improves held MAE over zero force by `17.02%`, but is `2.29x` worse
than the existing source energy derivative. At parent level it is 0/32
against the source and 31/32 against zero. Fit and held errors are close while
fit itself remains poor, identifying representation/readout underfitting
rather than conventional overfitting.

This force checkpoint must not initialize the Hessian/response head. Longer
training, Graphformer unfreezing, and validation/Test100 access are not
authorized from this result. A successor must expose intermediate G3D edge
messages, attention values, or explicit vector channels rather than relying
only on the final invariant atom state. Density response should read the
density branch before it is summed with element and distance embeddings.

Artifact:
`/home/shenwei01/xzh_node02_20260724/artifacts/graphformer_frozen_force_attention_pilot_v1_20260729`.
Formal summary SHA256:
`67fc46cc85338097a5da6e9a8ad1aa77694dbb711bdeb9d454e67ebca6ef4886`.
Checkpoint SHA256:
`aef7e09a86005e53f07b2d6011fc566780835770235d72cb1f83041b4b2b0eec`.
Detailed report:
`docs/qm9_graphformer_frozen_force_attention_pilot_v1.md`.
GitHub backup commit:
`2925904f65c294570cdefd59a9628ffc26ec867f`.

### Graphformer single-parent capacity-only branch (2026-07-27 22:08)

A new branch tests only whether the fresh Graphformer can fit the complete internal PBE Hessian
of train parent `0028399`. It does not supersede or erase the formal hybrid relaxed-HVP
performance failure, and it does not retain the old speed gate:

```text
protocol_id: qm9_graphformer_0028399_full39_capacity_only_v2
protocol SHA256:
7359095fc16e5378f1c7700747ac50ba1dbd84ce7f4e89e1c182123edb26f785
source checkpoint SHA256:
ca45fda1eedb9815e9cac26c23f95f0a60666f73c89d4a0e967803ddc4690a45
asset registration SHA256:
3286b301fd21d520e6db530b486bf7229a2732883b054768c6ab2269a652c1a0
scope: 0028399/sample0 only
directions: complete 39-direction internal basis, all capacity_train
loss: Hessian only; lambda_E=lambda_F=lambda_rho=0, lambda_H=1
pass gate: full39 relative Frobenius <= 0.05
```

Every update accumulates exact parameter gradients over all 39 directions while parameters are
fixed, then updates once. Independent fresh-checkpoint arms cover energy readout, final block,
final two blocks and full Graphformer, each with AdamW, L-BFGS and damped GN/LM. A matrix-free
CGLS audit reports the Jacobian linearized residual; nonconverged values are labeled upper bounds,
not minima. A no-pass optimizer run cannot authorize an insufficient-capacity conclusion.

The asset builder reads the frozen train-only split and the explicit `0028399` label/PBE Hessian
only. It does not read stable5/train20 manifests. Stable5, train20 held directions, validation and
Test100 remain locked.

Node02 focused tests pass `16/16`. All preflight failures remain preserved.
The strict-density floor near `1e-5` was caused by the new runner omitting
process-wide default float64 before sample construction, not by the checkpoint,
data or frozen parameter scope. The corrected path reaches `9.87e-11` in 306
cycles/19.27 s. The one-direction readout-adjoint smoke has finite gradient
norm `3.2525`, exactly zero parameter change, KKT residual `9.26e-13`, and
adjoint relative residual `6.00e-15`.

Formal summaries/checkpoints now bind implementation and registered-asset
SHA256 provenance; failed density endpoints retain coefficients and curves.
The first fresh full39 run failed closed at final projection because the
hybrid HVP is returned on CPU and the new projector was on CUDA. The corrected
projection follows HVP device/dtype; a formal-projection no-update smoke passed
with gradient norm `9.3592`, zero parameter change and adjoint residual
`5.38e-15`. The corrected fresh full39 baseline then completed with relative
Frobenius `2.575980`, MAE `0.074523`, RMSE `0.226372`, symmetry max
`1.21e-5`, density residual `1.19e-10`, and response residual `2.17e-12`.
This reproduces the old untouched baseline and remains above the 5% gate.

The sequential capacity supervisor remains active on node02 as PID `3028160`.
The independent `energy_readout + AdamW` arm completed all 40 configured
updates. Its final and best relative Frobenius is `0.5790032917`, with
MAE/RMSE `0.0239202258/0.0508816965`. This is a 77.52% baseline reduction but
is still 11.58 times above the 5% capacity gate. It closed fail-closed with
`capacity_passed=false` and `status=max_updates`.

The final AdamW checkpoint SHA256 is
`cf3a86db4fc3b71d9d209b72128a2785b83808c737dc0b906bc36aadd6392a53`;
summary SHA256 is
`4b2c465dd37e3d3b9960e55257c09ec75de174fad4870db7afe8bcda02a2ff50`.
Total wall time was 42.34 h, peak GPU allocation was 62172.97 MiB, and maximum
RSS was 75668.78 MiB.

The supervisor independently restarted `energy_readout + L-BFGS` from the
fresh checkpoint. At 2026-07-30 10:44 Asia/Singapore it had completed update
8 with relative Frobenius `0.8579292655` and MAE/RMSE
`0.0324695286/0.0753931750`; update 9 was at direction 28/39. L-BFGS is 15.03%
better than AdamW at equal update 8 but has not exceeded the completed AdamW
arm. Numerical residuals remain strict. No `CAPACITY_PASSED_FROZEN.json`
exists. Stable5, train20, held directions, validation and Test100 remain
locked.

Detailed protocol and current hashes:
`docs/qm9_graphformer_0028399_full39_capacity_only.md`.

### Clean rebuild analytic audit decision (2026-07-27 18:01)

This section supersedes the 13:12 clean-rebuild runtime status.

- The fresh train-only Graphformer EGF baseline completed exact fixed step 12,330. Checkpoint
  SHA256 is
  `ca45fda1eedb9815e9cac26c23f95f0a60666f73c89d4a0e967803ddc4690a45`;
  runtime batch telemetry is `(4,8,32)`, and no validation/Test100 selection was performed.
- Two structural second-order defects were isolated and repaired for the audit path. Classical
  Hartree, nuclear attraction and nuclear repulsion are now assembled from the untransformed
  physical-basis scalar energy. Natural reparametrization has an explicit native-eigh
  second-order audit mode; the historical custom mode remains the stable first-order default and
  is now forbidden by the analytic capacity runner.
- The evidence chain on frozen train parent `0028399` is:
  - physical-basis classical scalar curvature versus scalar FD relative error `5.23e-7`;
  - analytic density response versus strict endpoint coefficient secant relative L2 about
    `4e-6`;
  - response correction versus relaxed-minus-fixed FD relative L2 about `2.4e-7`;
  - corrected analytic HVP versus strict reoptimized force FD at `h=1e-4 Bohr` relative L2
    `3.34e-7`;
  - parameter-gradient versus parameter FD relative error `3.84e-4`;
  - maximum audited projected density-gradient norm `9.94e-11`.
- Formal full39 is finite and nearly symmetric:
  Hessian MAE `0.07452`, RMSE `0.22637`, relative Frobenius `2.57598`,
  and antisymmetric/symmetric Frobenius `1.203e-5`. This is the untouched baseline, so it does
  not satisfy the later `<=0.05` capacity target.
- Correctness, density, parameter-gradient, symmetry, full39 and solver-selection gates pass.
  Performance fails: estimated one-probe training step `111.97 s`, same-hardware legacy
  four-direction speedup only `0.882x`, full39 `1601.67 s`, peak GPU allocation `67.86 GB`,
  max RSS `77.93 GB`.
- Dense direct is the only strict solver with a parameter adjoint. PCG, MINRES and
  deflated-PCG fail the registered residual/response gates.
- Formal audit summary:
  `/home/shenwei01/xzh_node02_20260724/runs/graphformer_hybrid_relaxed_hvp_rebuild_v1/audit_0028399_formal_eigh_physical_classical_v1/summary.json`,
  SHA256
  `d38988ca3463dded5cec854ff449600055e783405d96bb586d010e56156a7a47`.
- The formal decision is `passed=false`,
  `single_parent_capacity_training_allowed=false`,
  `curvature_corrector_evaluation_required=true`, and
  `stable5_train20_allowed=false`. No capacity, stable5 or train20 training was launched.
- Final code, audit-chain and immutable-asset checksum lists are registered under
  `/home/shenwei01/xzh_node02_20260724/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/provenance/`.
- The next branch should evaluate an explicit scalar energy-conserving graph curvature corrector
  with train800 E/F replay. This analytic path remains a small-sample correctness oracle, not the
  training engine.
- Node02 paths only were used. `/scratch` was not accessed. Validation and Test100 remain unread.

### Clean rebuild latest status (2026-07-27 13:12)

This section supersedes the 10:58 status below.

- The final dataset-registration `split` `NameError` is fixed and covered by focused tests. Only
  finalization was rerun; the completed 3,200 checkpoints, labels and transforms were not
  regenerated.
- The frozen dataset manifest has SHA256
  `9506bf6573ecdea705555d204c389ce7508735baccc3c13533a7b23c623370ed`: 800 train parents,
  3,200 finite labels, 39,456 registered SCF-expanded samples, empty validation and empty Test100.
- GPU4PySCF environment validation, CuPy arithmetic and H2/PBE analytic Hessian pass. The
  registered environment package/hash list SHA256 is
  `3928950929fee10dd9bb7e885759c081e03eac3fa6cf12164a86a6d3da9a9d3b`.
- `0028399` GPU versus CPU PySCF Hessians agree to relative Frobenius `1.41937e-4`; the measured
  GPU reference run is `36.8x` faster. The formal train20 manifest has 20 successes, zero
  failures, and SHA256
  `acc5d00a984f17f73043aafcc5954fba7722537ab55c82f803a64f99a61f30c1`.
- During the first CPU cross-check PySCF wrote CPHF scratch groups into the source checkpoint.
  The mutated copy and failure record are preserved; removing only those newly added groups
  restored the original semantic checkpoint payload. The fixed helper disables checkpoint writes
  after SCF restoration and verifies source byte hashes before/after each Hessian.
- New stable5/train20 parent and complete-direction manifests are frozen. Direction-manifest
  SHA256 values are
  `e1d456e8c5dced55461a3067af3e1eaa158422e9551e982a364ba63b4f75eb1b` and
  `1ea9fe04a971b5998620d49f5e68567938ab4f505f17636fb2065085b8c73804`.
- A baseline launch was rejected after runtime telemetry exposed a Lightning single-rank DDP
  loader reconstruction bug: configured batch 4 became runtime batch 1. The run stopped at step
  431, has no registered checkpoint, and is preserved as a failed run.
- The fixed config sets `trainer.use_distributed_sampler=false`; a separate two-step smoke
  observed runtime `(4,8,32)`. The formal fresh baseline restarted from step zero on node02 GPU 7
  at 13:09 and will register only exact step 12,330 with the same runtime batch tuple.
- Analytic relaxed-HVP, capacity, stable5 and train20 training remain locked. Validation and
  Test100 have not been accessed.

### Clean rebuild runtime status (2026-07-27 10:58)

The frozen train800 electronic-structure and label stages are complete:

```text
Kohn-Sham: 3200/3200 .chk, verified, wall 31:28:46
labels: 3200/3200, wall 13:56:01
force check: 3200/3200 finite, 800 parents x 4 samples, failures 0
maximum force norm: 0.13713889631457066 Ha/Bohr
```

The corrected transform completed `3200/3200` at 10:52 with exit status 0. Split writing and
dataset-statistics generation also completed. Final dataset registration then stopped with
`NameError: name 'split' is not defined` in
`scripts/prepare_qm9_train800_rebuild_assets.py:297`; consequently
`train_only_dataset_manifest.json` does not yet exist and downstream baseline/HVP work remains
fail closed.

The first train20 PBE Hessian attempt remains a registered failure: all 20 entries failed
immediately with `ModuleNotFoundError` before GPU4PySCF was installed. The failed manifest and
registration were preserved as
`manifest.failed_20260727T101819.json` and
`manifest.registration.failed_20260727T101819.json`; they are forbidden inputs. The supervisor
validates `20 successes / 0 failures / finite`, and otherwise fails closed.

GPU4PySCF is now installed in a separate node02 environment:

```text
environment: /home/shenwei01/xzh_node02_20260724/envs/gpu4pyscf-cuda12-py311
gpu4pyscf-cuda12x: 1.7.4
gpu4pyscf-libxc-cuda12x: 0.8.1
cupy-cuda12x: 14.1.1
pyscf: 2.13.1
runner: scripts/run_gpu4pyscf_node02.sh
```

CuPy device arithmetic and a GPU H2/PBE analytic Hessian smoke both pass. The `6x6` Hessian is
finite with symmetry max abs `0.0`; its warmed GPU kernel time was `0.832 s`. The main project
environment remains unchanged at PySCF `2.4.0`, without CuPy or GPU4PySCF. Installation provenance
is registered at
`artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/provenance/gpu4pyscf_environment_20260727.json`.
The supervisor has been updated to call the dedicated runner for train20 PBE Hessians, but has not
been restarted while final dataset registration is failed. No baseline checkpoint, analytic HVP
audit, capacity run, validation access, or Test100 access exists yet.

### Graphformer relaxed-HVP clean rebuild branch (active, 2026-07-24 22:06)

The unrecoverable original-A checkpoint and old stable5 direction assets are permanently closed
for this experiment. A new branch is being rebuilt from a frozen train-only parent set and must
never be described as original-A or reuse its hashes:

```text
protocol_id: qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1
branch_id: graphformer_hybrid_relaxed_hvp_rebuild_20260724
method: 解析密度/KKT响应的hybrid relaxed-HVP
protocol SHA256: 06a1b0b232bd9bda45fdf94ffe05454845236e9fcedaac4af3d65238ea2f1bf9
node02 root: /home/shenwei01/xzh_node02_20260724
```

The only adopted old identity record is the readable frozen train800 CSV, SHA256
`882a0f1da1ee74575396731156cb4b6c191b456fcddca8681db48910677f90e9`.
It contains 800 parents and sample IDs 0--3, with empty validation and Test100. Old label
payloads were unreadable on degraded BeeGFS, so the branch keeps the same frozen parents and
deterministic geometry protocol but recomputes all 3,200 PBE/force labels under a newly frozen
identity. The 800 source QM9 XYZ files were copied from local AFS into node02 `/home`; their raw
manifest SHA256 is
`a009906f2545d300319dee149f7c44ab245cb6cd2c36dcbdec8bbb54a6973746`.

The active node02 label job is:

```text
PID: 3144255
launcher: scripts/launch_qm9_train800_fresh_labels_node02.sh
dataset: /home/shenwei01/xzh_node02_20260724/data/QM9PBEForceRandom1000Train800RebuildV1
run: /home/shenwei01/xzh_node02_20260724/runs/graphformer_hybrid_relaxed_hvp_rebuild_v1/labelgen
workers: 20 PySCF processes x 1 thread
```

The electronic-structure, label, force-check, transform, split, and statistics stages have now
finished. The launcher stopped only in final manifest registration at the `split` `NameError`
described above. Do not delete or regenerate the completed 3,200 `.chk`, label, or transformed
payloads.

The new EGF baseline is random-initialized and train-only. It uses the frozen force-weight-1
configuration, seed `676368232`, effective batch 32, and selects exactly step 12,330 without
validation. On node02 it will use the single currently available physical GPU 6 with batch 4 and
gradient accumulation 8. Its launcher refuses a busy GPU, an existing checkpoint, a nonempty
validation/test split, or a rebuilt sample count other than 39,456.

The branch orchestration is
`scripts/launch_qm9_graphformer_hybrid_rebuild_node02.sh`. It waits for the exact train20
sample-0 checkpoints, creates new PBE analytic Hessians, registers stable5/train20 parent and
complete `3N-6` direction manifests with new SHA256 values, runs the fixed-step baseline, then
runs the `0028399` audit. It cannot open stable5/train20 capacity training until the single-parent
capacity summary passes Hessian relative Frobenius `<=0.05` and E/F regression `<=5%`.
The original supervisor PID `3753337` is no longer active. The current post-recovery source
snapshot-list SHA256 is
`93f98c387cb664c52299e8e91c3f8b0c0bbcaac7bd6aeaa7be87cb3e5ea7c017`;
the prior snapshot list remains preserved with a timestamped provenance name.

Training uses one fresh **unnormalized** internal-basis Rademacher `+/-1` probe each step. This is
the unbiased internal Hessian Frobenius-squared estimator; unit normalization would change the
estimator and is forbidden. Direct, PCG, MINRES and deflated-PCG are compared for strict forward
response. At present only dense direct has an implemented implicit parameter-adjoint backward.
The audit fails closed if a faster strict solver lacks that training-grade backward.

Changed files for this branch include:

```text
configs/audit/qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1.yaml
configs/ml/data/qm9_pbe_force_random1000_train800_rebuild_v1.yaml
configs/ml/experiment/str25/qm9_pbe_force_train800_egf_rebuild_v1.yaml
mldft/ml/data/datamodule.py
scripts/prepare_qm9_train800_rebuild_assets.py
scripts/launch_qm9_train800_fresh_labels_node02.sh
scripts/launch_qm9_graphformer_rebuild_baseline_node02.sh
scripts/qm9_train800_pbe_hessian_references.py
scripts/prepare_qm9_graphformer_rebuild_manifests.py
scripts/qm9_graphformer_analytic_relaxed_hvp_audit.py
scripts/launch_qm9_graphformer_hybrid_rebuild_node02.sh
```

Static checks pass, and the focused density-response, internal-direction, capacity-training and
train-only datamodule suite passes `47/47`. No validation or Test100 record has been read.

### Graphformer analytic relaxed-HVP refactor (2026-07-24 21:10)

The single-parent Graphformer training path has been reworked to remove the four displaced
direction density optimizations from each step. It now samples one fresh Rademacher vector in the
complete orthonormal `3N-6` internal basis, strictly relaxes one center density, solves
`G_y y_v=-G_R v`, and constructs `H_relaxed v=L_RR v+L_Ry y_v`. Parameter backward uses an
implicit linear-solve adjoint. There is no fixed-density, stale-density, detached-HVP, or
force/Hessian-head fallback.

The PySCF boundary is semianalytic: directional second integral derivatives are finite
differences of complete first-derivative bundles, while density/model response, KKT solve, HVP
assembly, and parameter gradients remain graph-connected. The strict displaced-density
force-difference path is retained only as the correctness oracle. A parameter-step density
response predicts the next center density, but every next step still strict-corrects to projected
gradient below `1e-8`.

Focused static/tests pass `44/44`. The formal node02 preflight failed closed in `6.00 s` because
this exact file is absent:

```text
/home/shenwei01/xzh_node02_20260724/artifacts/original_A/last.ckpt
required SHA256:
e6516b04917a9dfaae3c4d960b10d77c768f3e5e784f95290ff41c1202f9d9bc
```

The stable5 parent and direction manifests with hashes
`72ecaab022402fb59de405487e4259e33d3438691013c2fcb67ae2d3379d774e` and
`1d9d235719805b3bf298fe3baf783b1b17a262e352b4193388f0be4f9b65bd74` are also absent from
their node02 `/home` paths. The local `trained-on-qm9` checkpoint has a different hash and must
not be substituted. The preflight used no proxy, read no validation/Test100 records, and left no
process.

Do not start stable5/train20 until the exact assets are restored and the following `0028399`
gates pass: analytic-vs-strict-FD relative error `<=1e-5`, parameter-gradient FD error `<5%`,
full39 non-regression/symmetry, at least `10x` speedup, and less than `60 s/step`. If the
correctness gates pass but the two performance gates fail, evaluate an explicit
energy-conserving graph curvature corrector instead.

Read:
`docs/qm9_graphformer_complete_total_analytic_relaxed_hvp_refactor.md`.

### node02 local rehome (2026-07-24 16:15)

The user explicitly abandoned `/scratch` and selected node02 local storage for the next training
round. This supersedes the earlier node01-only operational default for this work. Do not read,
write, probe, or silently fall back to `/scratch`.

The active node02 root is:

```text
/home/shenwei01/xzh_node02_20260724
```

It contains a clean baseline clone under `github/structures25` and an exact current WIP copy under
`work/structures25`. Both are based on local/GitHub HEAD
`a6c4c0b0e73cad5d3b89dbf8d6194b274c6cc4b6`; only the WIP copy contains the accumulated
uncommitted force/Hessian work. The active environment is Python 3.11.15 with the locked CUDA
stack, PySCF, PyG, and tensorframes. Eight A100-SXM4-80GB devices are visible and a real CUDA
float64 operation passed. The latest internal-direction/complete-total focused tests pass
`23/23`.

The verified local workstation `_runtime` snapshot has been mirrored independently of BeeGFS to:

```text
/home/shenwei01/xzh_node02_20260724/runtime_parent/_runtime
```

The mirror is complete and content-identical: 144,455 regular files, 18,765,821,111 bytes, and
source/destination content-manifest SHA256
`8dc8edb789bd039aa553ce6fc3a2ed24898de822a1a28d53c2df8bf211698fb9`.
P1-410 has 1,640 `.chk`, 1,640 raw labels, and 1,640 cached labels; all 1,640
force labels and numeric arrays pass finite/shape checks.

P1-410 labels/checkpoints/references are present in the mirror. The local snapshot does
not contain the 40 GiB random1000 dataset, full-QM9 labels, or the hash-bound original-A/bound
step-20 Graphformer assets. In particular, local
`_runtime/models/train/runs/trained-on-qm9/checkpoints/last.ckpt` has SHA256
`9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09`; it is not the
registered original-A hash and must not be substituted.

Use:

```bash
source /home/shenwei01/xzh_node02_20260724/work/structures25/scripts/activate_qm9_node02_local.sh
```

The setup, paths, verified capabilities, missing assets, and start policy are recorded in
`docs/qm9_node02_local_rehome_20260724.md`.

Node02 verification completed:

- locked Python/CUDA environment and 268-package dependency check pass;
- current WIP source matches the local workspace over 1,106 files;
- focused HVP/internal-direction tests pass `23/23`;
- MLDFT EG/EGF/HVP model tests pass `9/9`;
- checkpoint save/resume test passes;
- P1 EG and EGF lambda=1 each complete a two-step GPU smoke; EGF logs finite force loss.

No formal training was launched. At the final resource check, five root-owned non-Slurm GROMACS
processes occupied GPUs 0, 3, 4, 5, and 6 even though Slurm reported node02 as idle. Only GPUs 1,
2, and 7 were free. Do not start an eight-GPU run until direct `nvidia-smi` confirms all intended
devices are free.

### Scratch integrity hold (2026-07-24 14:50)

The Graphformer relaxed-HVP refactor is temporarily held before another scientific run because
the shared BeeGFS filesystem is degraded. Storage targets 5 and 6 on `node_storage_3` are
`Offline / Good`; all other targets are online. Existing directories remain visible, but reads
and creates touching the unavailable targets can block in D-state. In particular, the remote
environment interpreter cannot currently be certified, and the original-A checkpoint full hash
did not finish within the bounded audit window.

The local source workspace passes `git fsck` and Python compileall. An exact code/config/test/doc
recovery copy is available on node01 local XFS:

```text
/home/shenwei01/xzh_recovery_20260724/structures25
```

Its checksum comparison covers 1,129 regular files and reports zero transferred differences;
remote `git fsck` and compileall also pass. It is code-only and includes current WIP, so it is not
a validated training release. Attempts to recover directly under
`/scratch/xzh/recovery_20260724` are incomplete and must not be used.

Three critical Graphformer artifacts remain fully readable and hash-valid: the stable5 manifest,
the train20 manifest, and the bound step-20 checkpoint
`c18d4352584379b2f3607b0057fa633305b0ee8994d705fef060d33852c0876b`.
No labels, checkpoints, references, or evaluation outputs were deleted or overwritten. Full
details and the recovery sequence are in
`docs/qm9_scratch_integrity_audit_20260724.md`.

Do not start training from `/scratch/xzh` until targets 5 and 6 are online, D-state tasks have
cleared, the original-A hash and Python environment pass, and the relevant data/reference trees
have completed bounded inventories.

The final audit process snapshot contained eight D-state entries, including five bounded
integrity/recovery probes that cannot consume their termination signals while the storage target
is offline. Do not add more `/scratch` probes or attempt to clear them by repeated kills.

### Graphformer relaxed-HVP reset (2026-07-24 08:30)

The current task has restarted Hessian fine-tuning from the untouched original-A Graphformer.
Read the new top section of `docs/qm9_complete_total_hessian_capacity_v1.md` before operating it.
The hard target is a complete-total relaxed-force secant HVP from one scalar

```text
E_total = E_Graphformer_kin_plus_xc + E_H + E_ext + E_nn,
F = -dE_total/dR,
Hv = -(F(R+h v)-F(R-h v))/(2h).
```

Every displaced density is independently converged below projected gradient `1e-8`; a constrained
implicit VJP supplies parameter response. There is no fixed-density fallback, detached evaluator,
force/Hessian head, validation read, or Test100 read.

Frozen artifacts:

```text
source checkpoint SHA256:
e6516b04917a9dfaae3c4d960b10d77c768f3e5e784f95290ff41c1202f9d9bc

stable5 directions:
/scratch/xzh/models/graphformer_relaxed_hvp/20260723/v1/directions/stable5/manifest.json
SHA256 1d9d235719805b3bf298fe3baf783b1b17a262e352b4193388f0be4f9b65bd74

train20 directions:
/scratch/xzh/models/graphformer_relaxed_hvp/20260723/v1/directions/train20/manifest.json
SHA256 a902bb554cf05393c14b316e091f182c19bf888317c42aec153da774f02af378
```

Real derivative audit job 4056 passed: graph-vs-fresh relaxed FD relative L2 `1.679e-9`;
parameter-gradient-vs-reoptimized-FD best relative error `2.163%`; implicit solves converged in
`3421/3468` iterations at relative residual below `3e-5`. Full 39-direction job 4058 passed the
symmetry gate with `asym/sym=2.921e-5`, while original-A Hessian relative Frobenius is
`2.72659885`.

Job 4059 is an invalid smoke only. It revealed that the old loop used stale post-update densities
at step 2 (`projected gradient=6.29e-2`) before scheduling a refresh. It was canceled and must not
be cited. The trainer now requires strict active-density refresh before every parameter-step graph
and hard-fails at `>=1e-8`; focused tests pass `27/27`.

The initial `lambda_H=1` job 4060 was canceled without a final full-Hessian result after a
20-step weight screen established that stronger curvature weighting was necessary. Jobs
4061--4065 tested `lambda_H={3,10,30,100,300}` independently from original-A. All pass strict
density stationarity but all fail the 5% Hessian gate. Their step-20 Hessian relative Frobenius
values are `2.74695/2.36091/2.16958/2.15708/2.18248`. Weight 100 is the best curvature result;
its energy error changes by only `+0.245%` and its force MAE improves slightly. Weight 300 breaks
the 5% E/F non-regression requirement.

Job 4066 was the only continuation fit:

```text
/scratch/xzh/models/graphformer_relaxed_hvp/20260723/v1/capacity/
stage1_one_parent_h100_resume20_to200_stable5_all_s180_lr1e-6_4066
```

It resumed the exact step-20 model and AdamW state from job 4064 and targeted cumulative step 200
with `lambda_H=100`. It was canceled after producing valid training logs through cumulative
step 25; no newer accepted checkpoint was written. The reliable recovery point remains the bound
step-20 checkpoint with SHA256
`c18d4352584379b2f3607b0057fa633305b0ee8994d705fef060d33852c0876b`. Its metadata binds the
original-A root hash, strict protocol, direction manifest, molecule, original E/F gate baseline,
and all loss weights. Checkpoints are written every ten cumulative steps. The resume preflight
fails on any provenance or loss-weight mismatch.

Job 4066 is closed and will not be extended. Do not launch stable5 or train20 unless a new
single-parent run passes Hessian relative Frobenius `<=0.05` and energy/force regression `<=5%`
after the storage integrity hold is cleared. All such later stages remain closed. Focused
derivative, direction, strict-density, resume-binding, and complete-total training tests comprise
29 passing tests; validation and Test100 remain unread.

The completed precursor milestone was the random1000 HVP100 direct-curvature
supervision experiment. It uses 100 frozen parents from train800 and never trains or selects on
Test100. Train100 branch labeling, val20 stability labeling, the 20-run single-seed screen, and the
15-run three-seed comparison are complete. Direct-HVP variant C at weight `1e-4` and reference
floor `0.01` passed both the preregistered three-seed Tier-1 gate and strict complete-total HVP
validation against matched-seed A baselines. Its five-molecule full-Hessian/vibration check then
completed but did not pass the frozen symmetry non-regression gate, so no candidate is promoted
and Test100 remains forbidden. Read
`docs/qm9_random1000_hvp100_curvature_supervision.md` before extending this work.

The frozen 370-task resume completed without loss of the existing 830 records; the formal train100
inventory now contains 1200/1200 summaries. Strict validation array 2967 completed all 144
one-run x one-direction x one-molecule tasks on node01; analysis 2968 promoted one frozen C run to
the final validation-only full-Hessian stage. Full-Hessian array 3112 and analysis 3121 completed
10/10 tasks with zero failures, but produced no validation-promoted candidate.

Operational constraint: unless the user explicitly names another node, all future project
computation must use only `node01`. Active HVP Slurm scripts now default to node01 and GPU arrays
must use at most eight concurrent tasks.

The active milestone is the complete-total capacity reset below. Labeled-parent exact HVP
completion passes only as a matrix-completion upper bound. The frozen spectral/operator train20 CV
and global geometry-scalar full-Hessian CV both fail unseen-parent gates. The subsequent anchored
local-scalar L0/L1 models also fail the stable5 fit-only gate at about `0.94` median relative
Frobenius. Unanchored same-scalar J2 then preserves E/F but fails curvature at `1.036` median.
Stable5 activation v3 and three-task v4 are complete and rejected. J5 tanh is the best joint arm
at Hessian median/P90/max `0.5136/0.7013/0.7778` with E/F ratios `0.5740/0.6885`, but its frequency
MAE is `537.8 cm-1`, mode overlap `0.568`, and imaginary-mode total `53` versus PBE `11`. Frozen
Jacobian/LBFGS diagnostics show no 5% capacity evidence for this representation. The first explicit
three-body angular local scalar also fails: its neural smoke barely changes the median, while an
exact linear feature-span solve reaches `0.2780` median and `0.3096` maximum with excellent E/F.
Adding 10,560 parity-even bonded torsion columns improves this to `0.1872/0.2175` median/max, but
the frozen 2,000-step CGLS solve is not numerically converged. Hash-bound v6b reaches
`0.1475/0.1684` median/max at 20,000 iterations but still has a `2.55e-5` relative normal
residual. The same-ridge float64 Gram/Cholesky solve closes at `2.73e-13` normal residual but only
reaches `0.1361/0.1557` median/max, proving a feature-span failure. Central-element-resolved v7
job 3779 improves stable5 median/P90/max to `0.07010/0.09704/0.09825`, with exact normal residual
`2.19e-13`, energy median error `2.20e-10 Ha`, force median MAE `3.65e-5 Ha/Bohr`, and scalar
symmetry at about `1e-12`; it still fails both 5% Hessian gates. Its vibration entry therefore
fails closed. Higher-resolution v8 improves to `0.05435/0.08675/0.08833` median/P90/max but still
fails; its CPU-offload and hash-bound parent-jet cache now support exact-autograd restart after
OOM. Nonlinear local random-feature scalar v9 is the first representation to pass all stable5
gates: Hessian median/P90/max `0.01174/0.02413/0.02833`, frequency MAE/RMSE
`8.19/23.40 cm-1`, mode overlap `0.9401`, and model/PBE imaginary totals `14/11`, with excellent
E/F and scalar symmetry. This is a five-fitted-parent capability ceiling only. Its frozen train20
test is now complete and fails: train HVP median/max is `0.00952/0.03548`, but held HVP
median/P90 is `1.7439/9.5859` and full-Hessian median/P90/max is
`1.1407/4.5739/10.1497`, with `0/20 <=0.20`. A diagnostic-only ridge path confirms that
regularization helps but is insufficient: the best held setting, ridge `1e-3`, reaches held
median/P90 `0.5228/0.8794` and full median/P90 `0.3648/0.5379`, while breaking the train HVP
gate and retaining `0/20 <=0.15`. Random-width reduction also leaves `0/20 <=0.15`. Increasing
geometry-only training coverage to 32 or the maximal internal complement lowers full median to
`0.332/0.395`, but the tail stays at `3.93/2.50` P90 and added training directions exceed the 5%
fit gate. The all-label full20 capacity upper bound is much stronger at median/P90/max
`0.0668/0.0873/0.1051`, with `20/20 <=0.15`. Thus v9 can represent useful 20-parent curvature,
but partial-direction training does not identify a transferable curvature subspace. Parent
generalization, train100 expansion, independent validation reuse, and Test100 remain unauthorized.

The newest train-only Stage-2.5 diagnostics are also complete. The scalar-Rayleigh `q(v)` run on
59 parents is rejected and has a source-checkpoint provenance mismatch, so it is not same-scalar
evidence. A stricter successor then selected 20 parents using only the pre-existing relaxed-vector
stability mask, recomputed base energy/force from the same epoch-9 EGF checkpoint that produced
the source relaxed-force secants, and trained one scalar correction against 44 full vector HVP
directions while holding out one direction on each parent. This removes the q-only
underdetermination and checkpoint mismatch, but still fails decisively: the only arm that passes
the train-direction gates (`ridge=1e-4`) has held-direction relative-L2 median/P90
`4.817/43.550`, `0/20 <=0.15`, and only `3/20` improvements over the source. The arm selected by
held median (`ridge=1`) improves `15/20` directions but has held median/P90 `2.022/24.814` and
fails the train gate. Energy, force, finite-value, and scalar-symmetry checks pass; direction
generalization does not. Therefore no replay, full-Hessian, validation, train100, replacement
label, or Test100 stage is authorized. Eleven vector-stable train parents remain untouched by
this fit and must be frozen before testing a genuinely lower-dimensional, physically structured
scalar curvature kernel.

### Train20 parent-heldout operator audit (2026-07-22)

Protocol:
`configs/audit/qm9_complete_total_hessian_parent_cv_bounded_operator_v1.yaml`, SHA256
`b30e74c02ec244fce095f725ba34c3c637412994a58de048f025a6f3540183da`. It uses five deterministic
natoms/composition-balanced folds over the same 20 train parents. Every fold refits feature
normalization and coefficients on 16 parents; the four held parents supply no fitting directions,
completion, or model-selection information. The seven external validation parents and Test100
are not read.

The source is strict complete-total original-A at `sample_id=0`: median/P90/max Hessian relative
Frobenius `2.733/3.153/3.568`, frequency MAE `2921 cm-1`, and 809 imaginary modes versus PBE's 33.
The strongest cross-parent signal is tanh-conditioned unbounded arm B: median/P90/max
`0.354/0.622/19.858`, frequency MAE/RMSE `276/502 cm-1`, 85 imaginary modes, and `19/20` parent
wins. It still has only `1/20` parents below `0.15`. Standard unbounded arm A is similar at
`0.320/1.091/22.059`. Correction caps of 1.0 eliminate the catastrophic tail but regress the
median to `0.865--0.890`; cap 0.5 regresses it to `1.793`. No arm passes.

Main output:
`/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/parent_cv_original_A_v1/analysis_v1`.
Summary SHA256 is `409fbe315e7999adbd9d78452517d189cb2bce2317955386c2c00b11372011d4`;
plot SHA256 is `72f14795d9b57ebf1700569abe8af75eb8eb1d8342e0c65c218bb2be84e7997c`.
The summary records `advancement_authorized=false`, `test100_accessed=false`, and zero Test100
evaluations. Jobs 3707--3713 completed on node01 in `0:32--4:02` with `1.30--2.71 GiB` maximum
RSS; read-only analysis job 3729 completed in seven seconds.

### Train20 global geometry-scalar parent CV (2026-07-22)

Protocol:
`configs/audit/qm9_complete_total_hessian_geometry_parent_cv_v1.yaml`, SHA256
`5c81212f80f4c396214e748d8e3b0fc7990117a85188ac216f2b4ed4907fc479`. G0 uses a zero-initialized
float64 Softplus geometry-scalar MLP, exact scalar E/F/full Hessian, and five 16-fit/4-held folds.
Held Hessians are read only after each fold's best checkpoint is frozen. G1 would add train800 E/F
replay, but the protocol permits it only after G0 passes.

All five fit16 medians are `0.0212--0.0285`, so the model can memorize the training parents. Held20
median/P90/max relative Frobenius is `1.7305/5.2744/17.6050`, with `0/20 <=0.15` and `15/20` wins
versus original-A. Force median improves to `0.0838 Ha/Bohr`, while energy median worsens
8.83-fold to `0.8140 Ha`. Frequency MAE/RMSE is `1251/1868 cm-1`, mean mode overlap falls to
`0.4911`, and predicted/PBE imaginary modes are `300/33`. All Hessian, energy, frequency, and
overlap gates fail.

Per-parent oracle rescaling still gives median `1.605` and `0/20 <=0.15`; a strict-fold A/B/G0
linear combination remains near median `1.56`. This is a response direction/subspace failure, not
an amplitude, cap, replay, or optimization-step problem. Therefore G1 was not launched.

Artifacts are under
`/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/geometry_parent_cv_v1`.
Analysis summary/plot SHA256 values are
`5b69d06920c71b9858543b173abc6ea865f9d5f292b9206482a25b83600e93b9` and
`40e4b2e096e84c0b80ea47357f87261853bb780db98a2385ff7d60f7bb117d7e`. Jobs 3730_0--4 each
completed on one node01 A100 in `33:39--34:11`, with `10.7--11.4 GiB` host RSS and approximately
`34.6--35.1 GiB` peak GPU allocation. Analysis job 3735 took `1:05`. The summary records
`advancement_authorized=false`, `test100_accessed=false`, and zero Test100 evaluations.

### Stable5 local-scalar capacity reset (2026-07-22)

Protocol `configs/audit/qm9_complete_total_hessian_local_scalar_capacity_v1.yaml` has SHA256
`012420b6b3a92fdef24ba3dfc151f01513dbb43e050514f35193ce10dfaa70e4`. It reads only stable5
parents `0028399`, `0031108`, `0132419`, `0031012`, and `0121249`; summaries certify zero opened
unselected-parent artifacts. Validation and Test100 remain unread.

At each `R0`, original-A is represented by the local scalar Taylor model
`E_A-F_A*dR+0.5*dR^T sym(H_A)dR`. A smooth anchored scalar correction owns its E/F/H through
autograd, so the new matrices are conservative by construction and E/F are preserved at `R0`.
The scope is explicitly a local reference-geometry Hessian model, not a global OFDFT functional.

Arm L0 is the existing typed pair/triplet/torsion scalar. Arm L1 is a new invariant distance
message-passing scalar with no self loops, smooth radial/cutoff features, three Softplus message
layers, and initial-function subtraction. Focused remote tests pass `14/14`. Code/protocol SHA256:

```text
cddbd1af71629ca84704a578785d675b70f84ca8499067b1fb4be7e6d84da160  local_message_passing_residual.py
f92ce66bb1264e1fb32a4072ff96f8eb6dc5185f7d6b8a4e6be6ecf5e1df0730  local_scalar_full_hessian_capacity.py
e63810deebb0aefc1eaf04f8ebe2e8df83ac8a0952cc0caf16d0e48b8afaca32  local_scalar_capacity.sbatch
012420b6b3a92fdef24ba3dfc151f01513dbb43e050514f35193ce10dfaa70e4  local_scalar_capacity_v1.yaml
```

Smoke job 3736 completes ten steps in `71.9/60.0 s` for L0/L1. Starting stable5 median is
`2.7268`; L0 reaches `2.6926`, while L1 reaches `2.4111`. E/F ratios remain 1.0, asymmetry is below
`3.3e-15`, peak GPU allocation is `1.10/1.89 GiB`, and no NaN occurs. Formal 5000-step job 3738
then completed and failed: L0 median/P90/max is `0.93969/1.01040/1.05517`; L1 is
`0.94353/1.01311/1.05787`. Both preserve source E/F and exact scalar symmetry, but neither is close
to the 5% fit-only gate. They selected their final step, with wall times `3:15:30/2:45:18` and
about `2 GiB` peak GPU allocation. Analysis job 3740 records
`parent_cv_design_authorized=false`.

The zero-force anchor is itself a tight capacity constraint. Projecting each target correction
onto the symmetric subspace with rigid translations/rotations in its null space gives a stable5
irreducible relative-Frobenius floor of median/max `0.04039/0.04178`. The audit artifact is
`local_scalar_capacity_v1/rigid_mode_audit_v1.json`, SHA256
`17b004d855649b48c8fecf242070fd657e37a00610b0762837284fbe8cd709d1`. The 5% gate remains
mathematically possible but leaves less than one percentage point of margin. Job 3738's much larger
observed error triggered the preregistered unanchored fallback instead of more anchored tuning.

That fail-closed branch has now been taken. Protocol v2 removes the zero-force anchor while keeping
one scalar owner:
`E_A-F_A*dR+0.5*dR^T sym(H_A)dR+C_theta(R)-C_initial(R)`. Energy, force, and full Hessian are all
autograd derivatives of this scalar. In job 3742, E/F improve but raw Hessian-task parameter
gradients are only `1e-5--5e-4` of E/F-task gradients, so PCGrad alone is insufficient and the v2
formal run is forbidden.

The superseding protocol is
`configs/audit/qm9_complete_total_hessian_local_scalar_joint_capacity_v2_gradnorm.yaml`, SHA256
`5bed589cd57bdc697f992ee350b5040aeb89ab8b0c6cfe802370376b7cf63f11`. It applies bounded
GradNorm before PCGrad. Smoke job 3743 reaches a balanced Hessian/E-F gradient ratio of `0.99` by
step 10, remains finite, uses `5.83 GiB` peak GPU allocation, and improves median energy/force
ratios to `0.653/0.871`; Hessian median is still `2.6687`, as expected for only ten steps. Focused
tests pass `18/18`. Formal 3000-step job 3744 wrote to
`local_scalar_joint_capacity_v2_gradnorm/formal/J2_unanchored_message_passing_gradnorm`; job 3745
performed the read-only gate analysis after completion. No other parent was opened.

Periodic trainer JSONL now includes full-set E/F source ratios, median E/F errors, asymmetry, and
selection score; previously it exposed only sampled E/F losses beside full Hessian metrics. This
observability patch does not alter running job 3744. Trainer/test SHA256 values at that patch point
were `492e3acaf6dbdc3f53d82510946762f982fb0d31fd8d82d063b85112c20b1310` and
`0ec834bc7fbed0456ba41b21fa6404159f794a5ea521110a629096e54916bfa3`; remote focused tests pass
`6/6`.

A contingent stable5 activation/Jacobian audit was prepared before submission. Protocol
`configs/audit/qm9_complete_total_hessian_local_scalar_activation_audit_v3.yaml` freezes a
same-seed beta-1 Softplus control plus beta-5 Softplus, SiLU, and tanh arms, changes no data or
target definition, and permits smoke only if J2 formally fails. It addresses the measured
four-to-five-order initial Hessian-task gradient deficit without confounding random initialization.
Protocol/wrapper SHA256 values are
`b510aaf7a8f3cc1f320565d5141af42006e88f9fb9b30a9abc2bc548c071c715` and
`e1f1c81cbc9b8cee94574039916c64467a47fc2eac61e78ddafb7da4260e3431`.
All activation paths pass finite second-derivative, symmetry, and parameter-gradient checks;
remote focused tests pass `10/10`. Updated model/trainer SHA256 values are
`e02300ae8915b8f4d2b235d5ce748567dd52bcbb2ef2b87bb162c70e50d92b30` and
`77e3a5d85018d9e249bb23cb711c511d80e37b298aa3335eac34c879690360a9`.
Smoke promotion is fixed before execution: finite diagnostics, better final joint selection score
and median raw H/E-F gradient ratio than the same-seed control, then at most two non-control arms.

Read-only job 3746 audited the final best checkpoint's energy, force, and Hessian parameter
gradients separately, both globally and per parent. This resolves the previous
combined-E/F logging ambiguity without updating parameters or opening new data. Script/wrapper/test
SHA256 values are
`6f3217598d97e66777f8f00066c3714df0a70033802863c90329eaebeeec0128`,
`6860a136f4de3b9fcb4067f35e14f15c2ec630be830af1adcd89a54b6a931edb`, and
`e3f626f2046712b8396f73d26e40e5bc7f5bc38a8bedc8f30669f1f982132a2e`; remote unit/static checks
pass `1/1`.

J2 formal job 3744 and analyses 3745/3746 are now complete. Best step 2500 passes energy, force,
and symmetry gates with source ratios `0.6637/0.9508` and asymmetry `1.05e-14`, but Hessian
median/P90/max is only `1.0363/1.1772/1.2257`; capacity fails and parent-CV remains forbidden.
Median correction-target cosine is `0.9263`, and an oracle scalar amplitude still leaves median
error `1.0046`, so this is a response-direction failure rather than only amplitude. Formal summary
and checkpoint SHA256 values are `9d6e4ee8497cf22e3fd3d48d085bbb0a56be6edd50441fcfff34967ed38af8b0`
and `213ad79ddf4095ddd0e09b72b6fb90aa7e48f8967766653d359d4a2a7a828dff`.

The three-task audit finds aggregate E/F/H gradient norms `103690/5332/17.4` and cosines
E-F `-0.9578`, E-H `+0.8923`, F-H `-0.9675`. It quantitatively explains why combined-E/F PCGrad
can preserve median E/F yet stall curvature. Audit JSON SHA256 is
`eceb1a75f460fd98fc56292926ca350bf88d534d9dfa77416bba676db6613036`.

Activation smoke 3747 selected J5 tanh and J3 beta-5 Softplus by the frozen same-seed rule. Their
joint scores/raw H-to-EF gradient ratios are `5.4603/3.95e-4` and `5.9638/2.59e-4`, versus control
`5.9663/1.75e-5`. J4 SiLU fails the joint-score condition. Formal array 3751 runs only J3/J5 on
stable5. Smoke summary/CSV SHA256 values are
`361e4fd32152a427d5f1f073faef6dd2055dcbb9c128fba375904e519b717207` and
`4571b87ead8254ecb48d7ea957b02faa1bd706310f43aed94fbbeedec8c7f5c4`.

Formal array 3751 is active on node01. At step 100, J5 tanh has Hessian median/max
`1.0928/1.2950` and E/F ratios `0.4157/1.0749`; J3 beta-5 Softplus has
`1.8907/2.4101` and `0.7383/1.7139`. Per-100-step wall times are 545/659 s, implying roughly
4.5/5.5 h for the frozen 3000-step runs. No promotion decision is allowed from this intermediate
point.

At step 200, J5 improves to Hessian median/P90/max `1.0051/1.0829/1.1266`, E/F source ratios
`0.8574/0.6978`, and asymmetry `2.48e-15`; it remains far above the 5% stable5 capacity gate.
J3's latest complete metric record is still step 100. Formal merge job 3753 is held on
`afterok:3751`. The merge reads only smoke-selected J3/J5, validates protocol/parent/Test100
provenance, and cannot authorize parent-CV unless every E/F/H/symmetry gate passes. Its
script/test/wrapper SHA256 values are
`51a0c83b346848f420d5cef0aec1f4f5f08cba0903506492413a3e0d8f2571e8`,
`e29ec384ce91b6adbfe63054a2834660ede5f44da55a8208bda579a8629177f7`, and
`5513352eb2f978a360e905f58ba0f629dfad29caa19fe741398f335382426aa3`.

The current trainer's PCGrad partition is `(energy + force)` versus Hessian. Because the separate
best-J2 gradient audit found strong E-F and F-H conflicts, an optional three-task E/F/H GradNorm
and deterministic PCGrad path has been added without changing the default. Related node01 tests
pass `29/29`; no run has been submitted. Implementation/trainer/test SHA256 values are
`6aafee85a971812b6603c87370fdd8e355d23ae52ec6bb3330d7cc8b67c77bf0`,
`61c06bd64981ded51f6954d617e519a71fb2b015faa7500e722eb853db538a0b`, and
`61503aa48b40e91c543c6bb2419c9db8315bac7d2c40999824b24736a04fda3f`.
J5 step 300 remains its best joint checkpoint with Hessian median/P90/max
`0.8889/1.0262/1.0726` and E/F source ratios `0.7627/0.5912`. At step 500 its Hessian is
`0.8266/0.9529/1.0081`, but energy regresses to `1.854` times the source error. J3 step 400 is
`1.0673/1.1966/1.2676` with E/F ratios `0.5416/0.9369`; its early descent has slowed. Neither arm
is currently promotable, and both continue to the preregistered 3000 steps.

At step 600, J5 refreshes its joint best to Hessian median/P90/max `0.7764/0.9266/0.9775` and
E/F source ratios `0.4968/0.6802`; J3 is `1.0162/1.1281/1.1921` with E/F
`0.9333/0.7659`. Both remain non-passing.

The failure-only successor is frozen in
`configs/audit/qm9_complete_total_hessian_local_scalar_three_task_pcgrad_v4.yaml`, SHA256
`853a3b0c451d2973fbe2a0d9c58f77f30e530d8766c241a7979d4c778e795d62`. It separates E/F/H
before GradNorm and PCGrad while retaining the v3 model, seed, targets, and gates. Its 20-step
smoke selects at most one formal arm. The Slurm entry refuses to run before v3 formal analysis or
when v3 passes; no v4 job is currently submitted. Smoke analysis/training wrapper/analysis
wrapper/test SHA256 values are
`b23149a34069171d1c4184f30f4bf4dc9674f64af39961fd16058f59dad49b9c`,
`d9892453b7fffcd1b7878c68865eaab590ba74c686897ef52bc76352ef6d3e25`,
`b2f915981d86b31f24d625d0bad00b887cda61b245693eb65ae17a1b31b621ac`, and
`577bb984426ad389bbaf88309e20f0bc5b602292da25ef0813d35eeefb116059`; node01 checks pass
`25/25`. Job 3754 is queued after v3 merge 3753 and conditionally submits only v4 smoke,
machine selection, one formal arm, and formal merge when v3 fails; it never submits parent-CV.
Generic formal wrapper/formal-submit/contingency SHA256 values are
`2f61ebf1f98fde7649e4b82814d5786a03c96e6131bcda53d92060e57a5b6e3b`,
`1c4f0c772d2b24a7926618c04180126071ee7c1ca8fd4383b9802f8bb0757a91`, and
`f6aa4c8601e67247a03b800ff73cf60c1c50e43d050db9d16f9b8bd4132695e8`.

Stable5 formal reports now include frequency MAE/RMSE, imaginary modes, and matched mode overlap
for original-A and every smoke-selected formal arm. The evaluator records manifest/hash and
Test100 false/zero and rejects nonzero Test100 counts. Evaluator/wrapper/test SHA256 values are
`42ab89cf20e70f407808aaa7ae79a047e372c8900764cce8c1c7287fdd95afda`,
`53541775e0626b3129568e98a193fef5c404cea30ac57f4f2fc835c24a31c662`, and
`9f2b970c28bf1fcbdb959219318d7021d3a820a0ac450f8dc0af004ae78c4ef4`; tests pass `8/8`.
Job 3755 runs after v3 merge, and the failure-only v4 chain invokes the same postprocessing.

Real-data baseline preflight 3756 completed in 6 s on frozen stable5. Original-A frequency
MAE/RMSE is `2203.3/3301.2 cm-1`, mean mode overlap `0.5644`, and model/PBE imaginary-mode totals
`157/11`. Summary/per-parent/summary-CSV SHA256 values are
`fdb2de3820fc0eb73a65949d231b7fdc959757336629b5d3f4c01c6517a9e261`,
`90b6ae2f42535891a0502d66c517371ec2d5b5edc8d9a6839db0bdd47beaf5ee`, and
`43673dc1bf88160678798b9ea04ce0bcb745cbfe6d9acb5b088b225b1f552382`.
Latest v3 records are J5 step 900 Hessian `0.7112/0.8603/0.9301` with E/F
`0.3072/0.7124`, and J3 step 800 `1.0000/1.0901/1.1425` with E/F `0.6717/0.7353`.

Before v4 launch, the trainer was corrected so separate-task GradNorm preserves Hessian warm-up:
the H target norm is `warmup * ||g_force||`, rather than immediately cancelling the warm-up.
Trainer/protocol/test SHA256 values are
`61c06bd64981ded51f6954d617e519a71fb2b015faa7500e722eb853db538a0b`,
`853a3b0c451d2973fbe2a0d9c58f77f30e530d8766c241a7979d4c778e795d62`, and
`b0decdb61ce93c1b589f588d2b29dbb462a7c258af7b771b4b857f9270533d70`; tests pass `25/25`.

At J5 step 1000, correction-target cosine/norm ratio/oracle amplitude medians are
`0.9742/0.9601/1.0103`, but oracle-amplitude Hessian error remains `0.6926`. Because the target
correction is `2.7268` times the PBE norm, reaching 5% requires cosine about `0.999832`; all five
parents remain at `0.584--0.924`. This is not a single hard-parent tail. J3 step 900 is similarly
plateauing at median/max `0.9933/1.1137` while preserving E/F.

The v4 protocol stage was added to the shared trainer allowlist after preflight found it would
otherwise reject the automatic smoke. A hash-checked protocol-load regression now covers this
path. Current trainer/test SHA256 values are
`61c06bd64981ded51f6954d617e519a71fb2b015faa7500e722eb853db538a0b` and
`b0decdb61ce93c1b589f588d2b29dbb462a7c258af7b771b4b857f9270533d70`; focused checks pass
`7/7`.

A full-batch LBFGS ceiling runner is ready but not submitted. Joint E/F/H mode can authorize the
next stage only through all existing gates; Hessian-only mode is permanently diagnostic and
cannot authorize parent-CV. It records full closure counts, best/last checkpoints, per-parent
arrays, resources, source hashes, and Test100 false/zero. Runner/wrapper/test SHA256 values are
`d5438cb4fee290b1de6dd80fd6978bd980f0934eeecb82bac8e4e00be95e4cd3`,
`2957aa8bdd896e34349e6a43c11d514642f9502a5cf0393e9d0e2e7e8393846e`, and
`366ac3567243c8c74c2282826e47c83910bf27089825cfe77281b0d7d27167e1`; tests pass `3/3`.
Do not launch it until v4 is decided and its source checkpoint/protocol are frozen.

Runtime-only LBFGS smoke 3757 froze J5 step 1100 and ran one full-stable5 joint iteration. It used
three closure calls, `4769.9 MiB` peak GPU, `1599.1 MiB` MaxRSS, and 72 scheduler seconds with no
NaN/OOM. Score changed `1.600303 -> 1.600232`, Hessian median
`0.686959 -> 0.686927`; this validates the runner but is not evidence of useful LBFGS convergence.
Snapshot/summary/metrics/checkpoint/per-parent SHA256 values are
`ad0a8ce0c2c6d5b4f1e25cdd2bd491a798d8c0ff1fff2408645aef3af5d68a98`,
`2cf6faf392e1fb03f42c427039d288cc3efdc1e802a1dcaf4fd7b66b5a1280f0`,
`4fa7634f02cc699516175e7f19d8d666db2ae8e251d5a3caba395e4d96426ecd`,
`d6c7b678d4d1fe290fe1d0c912794cf1f4169bbdef6acb5eb055cd976a1e934a`, and
`9c7a7aec14ed280d726b353e17e31883f7bdc0bdaf35215e9db0c86d288d290d`.
J5 v3 step 1200 is Hessian median/max `0.6760/0.8963`, E/F `0.2388/0.7202`, still non-passing.

The paired-v3 smoke thresholds in v4 now exactly equal the machine summary values
`5.963815683324032/5.460310807421687`; a `3--5e-7` hand-transcription difference was removed.
The superseding v4 protocol SHA256 is
`853a3b0c451d2973fbe2a0d9c58f77f30e530d8766c241a7979d4c778e795d62`.

A new stable5-only rigid-invariant scalar-jet audit rules out inconsistent derivative labels as
the capacity blocker. For the target correction `PBE - original-A`, the nearest symmetric Hessian
compatible with the rigid-projected target force has relative-Frobenius floor
median/P90/max `0.0004212/0.0004609/0.0004776`; allowing any consistent invariant force gives
`0.0003782/0.0004340/0.0004543`. Median correction net-force and net-torque defects are only
`7.64e-5` and `1.02e-5` in the audit's relative definitions. Therefore the unanchored 5% gate is
mathematically well inside the compatible scalar-jet subspace; the observed J5 plateau is a
second-order representation/Jacobian and multitask-optimization issue, not a sign, unit, rigid
motion, or force/Hessian consistency error. The audit opened only frozen stable5 artifacts and
records validation/Test100 false/zero. Outputs are under
`local_scalar_activation_audit_v3/jet_compatibility_audit_v1`; script/test/summary/CSV SHA256 values
are `e860ef12d80eed01765af6c0ef20b3c44a32b6a7b471fed2d13c5107405bb133`,
`01060803abbdd5e6e0bff211e825fc7ad26eba506bd30027f1595e4b15ad9777`,
`2be7bb4dccfce012fb9a3aaed0a97341be23171d3064d5f5c48ff53b25a2a23b`, and
`01d5515c6059ea4516c51f94eb44a04910a326c5f87ccfd624dd0f61362e27d1`.
Focused tests pass `2/2`.

Latest active v3 records are J5 step 1500 Hessian median/P90/max
`0.6571/0.7853/0.8673`, E/F `0.3673/0.7086`, and J3 step 1300
`0.9674/1.0365/1.0809`, E/F `1.3738/0.7025`. Neither passes; array 3751 continues to its frozen
3000 steps. Validation parents and Test100 remain unread.

A matrix-free stable5 Jacobian-range tool is ready for the post-v4 representation audit. It uses
centered float64 parameter-space `J*v`, exact scalar-autograd `J^T*u`, and CGLS without forming the
`6888 x 153889` Jacobian. Runtime smoke 3761 used the immutable J5 step-1100 snapshot for one
iteration and completed in `70.9 s` with `4.76 GiB` peak GPU and `1.45 GiB` MaxRSS. A threefold
FD-step reduction gives `J*v` relative difference `7.67e-8` and cosine
`0.9999999999999998`; there is no numerical blocker. The one-step median change
`0.68696 -> 0.68528` is only a runtime check. Do not interpret the Jacobian range until the
preregistered 30-iteration run is made from the checkpoint frozen after v4. Script/wrapper/test
SHA256 values are `9f58594a01daf56dcd3e2bca1dbb83e2da40fded55f8c8bba2a93c1e02fb95b7`,
`4a86b7d7ddaa8944f74489b453ef44f29703aca193625811aa05e1bff9356be5`, and
`c23d16649ccd0bfa9f75d4a0d30918eacbedd3929892ae6072659a0a677ba2ff`; clean smoke
summary/metrics/resource hashes are
`bfdc7e0886f4d25cef7309dd576b935a090af7212d84ce14e4085f14a4d16c1b`,
`d9ada41db91b1a28c4cc518157a686bf113265b627ee3b8b973441b60e06ae33`, and
`6867ee9d7887d30341f555f66c11b984de641a30cb56007ea1c139d92a7a580d`.

The v4 post-failure diagnostics are now automatically frozen rather than manually assembled.
After v4 formal analysis, the dependent wrapper exits on a pass. On a failure it creates a JSON
manifest binding the formal summary, unique failed arm, best checkpoint/step, exact stable5 list,
all source hashes, 30-iteration Jacobian settings, and joint/Hessian-only 100-iteration LBFGS
settings. The LBFGS runner requires that manifest and rejects any source or numeric-setting drift.
Hessian-only can never authorize parent-CV; even a passing joint result only produces evidence and
does not submit parent-CV. The three diagnostics run on node01 and never open validation or
Test100. Freeze/LBFGS/Jacobian tests pass `9/9`, and all wrappers pass `bash -n`. Freeze script,
LBFGS runner/wrapper, post-v4 wrapper, and updated formal orchestrator hashes are
`a643f20c826ec49db99fd73244bd31c5b92d31bc105b804ed6b5b9db47ad299c`,
`e44827f6b7943dac78ec371612f1159c70bae1226d7182cd049e9490269458c7`,
`4f346cf706b073e853ed29f6dafe7c5de5ae1cde2495267dfe04f0d20328bda0`,
`1cc4d9e27857c1ddaa4f2e2d3088bf0e9c65e8f5c463011bb2a8a21932509378`, and
`f3b4b92acd728c09be2ff258f800bdf0ef624b027f22fb357cbe43793f7588c2`.

A dependent read-only merge now validates and classifies all three post-v4 diagnostics. It plots
the Jacobian CGLS residual, both LBFGS Hessian trajectories, and joint E/F gates, and distinguishes
joint capacity, Hessian-only representation capacity with multitask failure, a local linearized
range pass with nonlinear optimization failure, or no 5% evidence under the frozen budget. Only
a complete joint gate pass can set `parent_cv_design_authorized=true`; the merge always records
`parent_cv_submitted=false`. Analyzer/wrapper/updated submitter/test hashes are
`c1bee20b863011670ef005480105e0a2c3b5ae8e4984c05a3e9793a2ef0d80fa`,
`985fdd9b57ffa9952e13b6b663e4e179b2a953fd1061e3e2169293803f4a1c53`,
`f7036c84afd839ebe66fcd8e7007adc4b2db9c335befb77c072bee9aac0c1ea7`, and
`10b8659bcd3bd9805e08c0fe592952818a8dc30782c4df5985f46437a586b418`; the
integrated suite passes `10/10`.

J5 v3 step 2000 is the latest joint best: Hessian median/P90/max
`0.5935/0.7499/0.8340`, E/F `0.3163/0.7204`. It remains almost twelve times above the median 5%
gate. Validation and Test100 remain unread.

The automatic diagnostic chain now also handles a v4 smoke result with no eligible formal arm.
In that branch it chooses the best failed v3 formal arm by the frozen joint score
`median_H + max_H + E/F gate penalties`, records that rule in the manifest, and launches the same
stable5-only diagnostics. Dynamic source paths are accepted only through subsequent hash checks.
Updated freeze/post/merge-wrapper/formal-orchestrator/test hashes are
`37515e928bab172aa59b0bd59e65ece237eed09d870d6748ccbdb9fabbc4af72`,
`5cd6174eaabb52a626bce94ca47799aaaca4da716840bbed854fdc302bb0d2f5`,
`2432e271cbda952beecc0c0536f4c5bc098b23a15788ce081cc3ae3e567fa72a`,
`c500f15da12d7345f0be1d5161fc4302a587cdc4a1dc06ff602e3046fb67f612`, and
`d81c08149a593ee9572f56126b1f4cd917f1e3d4463fa8ffb1ce2efcef6856fd`; the integrated
suite passes `11/11`.

Stable5 promotion now includes a hash-frozen vibration closure rather than stopping at matrix and
E/F gates. The post-v4 manifest preregisters frequency MAE `<=200 cm-1`, absolute total
imaginary-mode-count error `<=5`, mean mode overlap `>=0.8`, and all five required molecules. A
new node01-only wrapper evaluates original A, joint LBFGS, and Hessian-only LBFGS, after which the
read-only merge can authorize only a joint candidate that passes Hessian median/max `<=0.05`, E/F
ratios `<=1.05`, asymmetry `<=0.005`, and every vibration gate. It never submits parent-CV and
records validation/Test100 false/zero. Original A itself has frequency MAE/RMSE
`2203.3/3301.2 cm-1`, mean overlap `0.5644`, and model/PBE imaginary-mode totals `157/11`.
Updated freeze/analyzer/vibration-wrapper/post-wrapper/merge-wrapper hashes are
`165bbfabab6400d08ce2be07d824166565897ebab30307fddb16709a834fa2b2`,
`75f37111650c893520f02e28c97ffbaefa315d3d10a6cbb6fcb28de85ae64e0c`,
`9b9658046d3fdd63a41f5779a41b59f8da620b4bf56964b7fc41b0687eb64c78`,
`188520e95f1dd8b4edf85199a51b8ee0fc486c64a1458fd99b61fcb9aa43a870`, and
`919b8abeeffd7016729eee4ab94ac71af17b3ec2f48f7660b271d769d5779b4f`.
The expanded freeze/merge/LBFGS/Jacobian/vibration/three-task/complete-total suite passes `44/44`;
static and shell checks pass. No unseen parent has been opened.

Formal activation v3 completed in jobs 3751/3753/3755. J5 tanh selected step 3000 with Hessian
median/P90/max `0.513559/0.701311/0.777794`, E/F `0.5740/0.6885`, frequency MAE/RMSE
`537.79/786.46 cm-1`, overlap `0.5680`, and `53/11` model/PBE imaginary modes. J3 selected step
2900 at Hessian `0.778141/0.944311/0.991500` and is worse. Formal-analysis/vibration hashes are
`c7f0eb8fd06d0c4a7118874f33278aab12e2eec6b8c2383f0c3c7b2cba100595` and
`33e8f7e15669cb4d00800ec45f15550788f45599fc81b7b4f8f107ef26a4df28`.

Three-task v4 smoke jobs 3762/3763 did not improve either paired v3 score or step-zero Hessian, so
no formal was launched. The no-formal branch froze J5 step 3000 and completed diagnostics
3766--3771. Thirty CGLS iterations reduce the local linearized median only to `0.4778`, leaving
`93.99%` of the initial global residual. Joint LBFGS selects iteration zero; Hessian-only LBFGS
reaches `0.4394/0.6180/0.6788` but regresses energy ratio to `3.5847` and leaves frequency MAE
`536.31 cm-1`. Final diagnosis is
`no_five_percent_capacity_evidence_with_preregistered_local_diagnostics`. The immutable manifest
and final-analysis hashes are
`0e03b754ef2006496b89bdb0da2d616b2e54d8e832a090db87d385f8b82f0022` and
`152c3ed97a47a27c365f3579677a8c8af2b52673e8b3d9cf1ee7e078df6e2ad4`; validation/Test100 are
false/zero and parent-CV was not submitted.

The v4 smoke selector now ignores expected forward-only step-zero NaN placeholders when checking
training-diagnostic finiteness, while still rejecting every positive-step NaN. This is a reporting
correctness fix only: both historical v4 arms independently failed their accuracy conditions.

New code:

```text
configs/audit/qm9_complete_total_hessian_parent_cv_bounded_operator_v1.yaml
scripts/qm9_complete_total_spectral_operator_capacity.py
scripts/qm9_complete_total_spectral_operator_parent_cv.py
scripts/qm9_complete_total_spectral_operator_parent_cv_analysis.py
scripts/qm9_hessian_vibrational_metrics.py
scripts/slurm_qm9_complete_total_spectral_operator_capacity.sbatch
scripts/slurm_qm9_complete_total_spectral_operator_parent_cv.sbatch
scripts/slurm_qm9_complete_total_spectral_operator_parent_cv_analysis.sbatch
tests/test_qm9_complete_total_spectral_operator_parent_cv.py
tests/test_qm9_complete_total_spectral_operator_parent_cv_analysis.py
tests/test_qm9_hessian_vibrational_metrics.py
configs/audit/qm9_complete_total_hessian_local_scalar_joint_capacity_v2.yaml
configs/audit/qm9_complete_total_hessian_local_scalar_joint_capacity_v2_gradnorm.yaml
scripts/slurm_qm9_complete_total_local_scalar_joint_capacity.sbatch
scripts/slurm_qm9_complete_total_local_scalar_joint_gradnorm_capacity.sbatch
configs/audit/qm9_complete_total_hessian_local_scalar_activation_audit_v3.yaml
scripts/slurm_qm9_complete_total_local_scalar_activation_audit.sbatch
scripts/qm9_complete_total_local_scalar_gradient_conflict_audit.py
scripts/qm9_complete_total_local_scalar_jet_compatibility_audit.py
scripts/qm9_complete_total_local_scalar_jacobian_range_audit.py
scripts/slurm_qm9_complete_total_local_scalar_jacobian_range_audit.sbatch
scripts/qm9_complete_total_freeze_local_scalar_post_v4_diagnostics.py
scripts/slurm_qm9_complete_total_local_scalar_post_v4_diagnostics.sbatch
scripts/qm9_complete_total_local_scalar_post_v4_diagnostics_analysis.py
scripts/slurm_qm9_complete_total_local_scalar_post_v4_diagnostics_analysis.sbatch
scripts/slurm_qm9_complete_total_local_scalar_gradient_conflict_audit.sbatch
tests/test_qm9_complete_total_local_scalar_gradient_conflict_audit.py
tests/test_qm9_complete_total_local_scalar_jet_compatibility_audit.py
tests/test_qm9_complete_total_local_scalar_jacobian_range_audit.py
tests/test_qm9_complete_total_freeze_local_scalar_post_v4_diagnostics.py
tests/test_qm9_complete_total_local_scalar_post_v4_diagnostics_analysis.py
scripts/qm9_complete_total_local_scalar_activation_smoke_analysis.py
tests/test_qm9_complete_total_local_scalar_activation_smoke_analysis.py
```

## Complete-Total Hessian Capacity Reset (2026-07-17)

The active objective is now a three-gate reset: small-parent capacity ceiling, unseen-direction
generalization, then unseen-parent generalization. The target is to reduce strict complete-total
density-relaxed Hessian relative Frobenius from about `2.7` to about `0.1`; Test100 is frozen until
a candidate and selection rule pass the first two gates. The preregistered protocol is
`configs/audit/qm9_complete_total_hessian_capacity_v1.yaml`. Its train-only Stage-1 manifest is
`/scratch/xzh/models/complete_total_capacity/20260717/stage1_manifest.json`, with manifest SHA256
`2d838c64bee4772c518be1ac9f786c2cb84175175d291a469f15772b7c61cebc` and source split SHA256
`8d5fc057a0ff6af2f7d994db7f6517145c00c745416b015caa3dd9afb57f1c7a`.

The initial v1 Stage-1 parents were `0028399`, `0050129`, `0031108`, `0132419`, and `0031012`.
This historical list is superseded by stable5-v2: the pre-fit curl gate excludes `0050129` and
freezes `0121249` as its replacement. The active list is therefore `0028399`, `0031108`,
`0132419`, `0031012`, and `0121249`; all are train100 parents with all four existing branch-audit
directions stable. No validation parent and no Test100 record is used for capacity fitting. New
implementation files are:

```text
mldft/ofdft/complete_total_training.py
scripts/prepare_qm9_complete_total_capacity_stage1.py
scripts/qm9_complete_total_capacity_train.py
scripts/qm9_complete_total_capacity_analysis.py
scripts/slurm_qm9_complete_total_capacity_smoke.sbatch
scripts/slurm_qm9_complete_total_capacity_one_parent.sbatch
scripts/launch_qm9_complete_total_capacity_stage1.sh
tests/ofdft/test_complete_total_training.py
docs/qm9_complete_total_hessian_capacity_v1.md
```

The first audit found a definition mismatch in the old direct-HVP route: historical training
differentiated only learned `kin_plus_xc`, while formal acceptance differentiates the scalar sum of
learned, Hartree, external and nuclear-repulsion terms after strict density relaxation. The new
trainer owns this complete scalar total energy and derives force from it. It records strict density
refreshes, total energy/force, force-secant HVP, scalar relaxed-energy directional curvature,
parameter-gradient norms/cosines, checkpoints and strict full/partial Hessian metrics.

One-direction job evidence on train parent `0028399` is currently decisive:

- detached block-coordinate pure-HVP update: relative column error `3.43132 -> 3.43938`;
- 10-step differentiable density unroll: `3.43132 -> 3.43949`;
- retaining Lagrange-multiplier parameter response alone: `3.43132 -> 3.43938`;
- pure scalar relaxed-energy curvature update: `3.43132 -> 3.39182`, while strict directional
  energy-curvature error decreased `2.57080 -> 2.54153 Ha/Bohr^2`.

Thus the scalar envelope-theorem gradient has passed a strict re-relaxation direction check, but
the detached force-secant HVP parameter gradient has not. A new matrix-free constrained implicit
VJP now passes an analytic unit test and a strict real-molecule direction check. At `3e-5` PCG
tolerance, two real endpoint solves converged in at most 1467 iterations and reduced relative
column error `3.43132 -> 3.39250`. The opt-in 8-probe Jacobi preconditioner made convergence worse
and is disabled by default.

Scalar directional curvature is an audit and capacity-training signal; it is not reported as a
vector HVP. Pure-Q job 3134 sustained improvement through 25 steps (`3.43132 -> 2.72905`) but
worsened energy error (`0.12042 -> 0.14388 Ha`). Equal-weight E/F/Q job 3140 improved energy,
force, and curvature simultaneously in one step: energy error `0.12042 -> 0.11434 Ha`, force MAE
`0.11607 -> 0.10986 Ha/Bohr`, and relative column error `3.43132 -> 3.23058`.

The training graph now separates stationary envelope E/Q derivatives from implicit-response F/HVP
derivatives. Stage 1 has not passed, no five-parent fit has started, and Stage 2/3 remain disabled.
Balanced implicit smoke 3142 completed on node01 and improved energy error
`0.12042 -> 0.11279 Ha`, force MAE `0.11607 -> 0.10946 Ha/Bohr`, and relative Hessian-column error
`3.43132 -> 3.20836`. The jump-host VPN briefly became unreachable and was recovered without
replacing the completed job. Ten-step balanced implicit run 3143 then improved energy error to
`0.03452 Ha`, force MAE to `0.07015 Ha/Bohr`, and relative column error to `1.88811` with monotonic
strict metrics. Scalar-curvature continuation 3144 reached `0.72395` at cumulative step 30;
balanced E/F/Q/H job 3145 reached `0.65760` at step 20.

The capacity trainer now checkpoints independent replay/HVP AdamW states and supports alternating
updates. Job 3150 validated this path (`0.65760 -> 0.60645`) but showed that response-solve cost is
too high to use it before the one-column ceiling is known. Pure-HVP job 3152 then reduced the same
strict column from `0.65760` to `0.48034` in ten updates, with density gradients near `1e-10`,
energy error `0.00813 Ha`, force MAE `0.05262 Ha/Bohr`, and wall time `29:11`. An opt-in
Frobenius-aligned RMSE HVP loss is implemented; controlled one-step job 3153 reached `0.62556`.
Longer runs changed the optimizer decision: L1 job 3154 reached `0.38114` at step 50 but regressed
energy error to `0.09479 Ha`, while L2 job 3155 reached `0.40002` already at step 40 with energy
error `0.00481 Ha` and force MAE `0.05227 Ha/Bohr`. L2 continuation 3157 completed step 60 at
relative error `0.22581`, but energy error and force MAE regressed to `0.05664 Ha` and
`0.06750 Ha/Bohr`. Pure L2 continuation 3162 completed through step 80. Its best strict point was
`0.16595` at step 78; step 80 rebounded to `0.17086`, with energy error `0.02795 Ha` and force MAE
`0.08884 Ha/Bohr`. It remains a capacity upper bound, not a promotable model. Job 3165 preserves
the step-80 optimizer state and lowers the HVP learning rate to `1e-6` for 20 bounded steps; no
five-parent expansion is permitted unless the one-column gate is resolved.

`GaussianLayer` no longer forces float64 distances through `x.float()`. On an unchanged checkpoint,
job 3156 kept relative Hessian error effectively fixed (`0.480373 -> 0.480357`) but reduced
relaxed-energy/force curvature closure error about 850-fold (`1.67e-3 -> 1.97e-6 Ha/Bohr2`). This
fix removes a numerical derivative inconsistency; it does not explain the model's large Hessian
error. Combined GBF and capacity tensor tests pass `21/21`, with static Python and shell checks.

Zero-LR gradient audit 3158 records per-loss norms, cosine conflicts, and per-module squared-norm
fractions in `training_curve.csv`. At L2 step 40, energy/force/Q/HVP gradient norms are
`1079.55/275.40/524.47/23.15`; HVP conflicts strongly with Q (cosine `-0.795`) but aligns with force
(`+0.239`) and is nearly orthogonal to energy (`+0.015`). The balanced path therefore drops Q and
alternates independent E+F+Euler replay and L2-HVP optimizers. Job 3159 is active for 20 updates of
each type from the step-40 checkpoint. Pure L2 job 3157 reached `0.22581` at step 60, with failed
energy/force tradeoff as recorded above.
Response-tolerance control 3160 is rejected: `1e-3` cut PCG iterations from about `1680` to `426`
but gave step-42 relative error `0.39445` versus strict-tolerance 3157's `0.38053`, with only modest
wall-time reduction. Training remains at `3e-5`. A tested sparse replay schedule is now available;
job 3161 compares 30 HVP plus 10 replay updates against equal 20/20 job 3159 from the same step-40
checkpoint. Capacity tests pass `12/12` after this addition.
The completed final comparisons are: 3159 `0.27558` relative column / `0.03852 Ha/Bohr` force,
3161 `0.22755` / `0.05471`, and gradient-scale-matched joint job 3164 `0.35059` / `0.03520`.
Replay protects force but none reaches the curvature capacity gate.

Default-off two-task PCGrad is implemented and passes complete-total tests `14/14`. Real one-step
smoke 3166 measured replay-vs-HVP gradient cosine `+0.183`, did not trigger projection, and reached
`0.39695`; PCGrad is not the current bottleneck. Pure job 3165 continued the historical AdamW
state to step 100 and reached `0.14712`, energy error `0.02389 Ha`, and force MAE
`0.08617 Ha/Bohr`. Resetting Adam history did not improve the plateau: a `3e-6` one-step probe
worsened to `0.33382`, and reset `1e-6` stayed near `0.151`. Job 3170 preserves the step-100 state
and runs the bounded pure-HVP upper bound to cumulative step 200.

Residual decomposition at pure-HVP step 80 gives predicted/reference HVP cosine `0.98566`, but
optimal scalar rescaling still leaves `0.16179` relative residual. Translation leakage is only
about `1e-6`, and the remaining error spans several atomic coordinates. The source Graphformer has
768 channels and four G3D layers. The immediate diagnosis is therefore an optimization/multitask
constraint problem, with functional-kernel capacity still an open possibility, rather than a
simple Hessian scale, rigid-motion, float precision, or density-convergence artifact.
Read-only aggregation now includes per-array cosine/scale/orthogonal and rigid-sum residual
diagnostics. The 18-run artifact at
`/scratch/xzh/models/complete_total_capacity/20260717/analysis_one_column_v1` contains 119 residual
rows plus CSV/JSON/PNG summaries and certifies zero Test100 access. The residual helper test passes
`1/1`.
Stage 1 remains failed; no multi-direction/five-parent work and no Test100 access are allowed until
a one-parent complete Hessian can meet the 5% capacity gate.

The bounded pure-HVP upper bound subsequently reached one-column relative Frobenius `0.09219367`
at step 200, with unacceptable energy/force regression. Preserving its optimizer and continuing
20 updates at `3e-7` reached only `0.08866575`; this confirms a practical plateau rather than a
reason to continue percent-level direct-HVP tuning. Geometry-kernel capacity audits then separated
the failure: a 30-configuration pair-RBF scan bottoms out at `0.05374092` with condition number
about `9e15`, whereas a conservative scalar three-body RBF-Legendre residual reaches
`0.01360579`, condition number `551`, zero anchor energy/force to `3e-15`, and finite symmetric
second derivatives. Job 3201 reproduced `0.0136057917` through the actual strict complete-total
density-relaxed evaluator, with no change to anchor energy/force and zero Test100 access.

The new reusable component is
`mldft/ml/models/components/three_body_geometry_residual.py`; audit/launcher/test files are listed
in `docs/qm9_complete_total_hessian_capacity_v1.md`. The strict step-200 full 45x45 baseline job
3202 completed at relative Frobenius `1.28832`, despite its trained first column being `0.09219`.
A shared-element three-body scalar kernel reached a structural floor of `0.05763` at design rank
690. Adding a parity-even four-body torsion scalar residual crossed the complete one-parent gate:
the selected stable configuration reaches `0.0467975`, MAE/RMSE `0.002231/0.004112 Ha/Bohr2`,
and `asym/sym=4.58e-5`. Anchor energy/force residuals remain below `2e-8` after the sequential
fit. Pair, three-body and four-body components are all scalar-energy terms; no force head exists.

The one-parent sub-gate is therefore passed. Jobs 3221--3224 are active on node01 for the other
four frozen train parents. Stage 1 as a whole remains failed until one shared scalar model or
coefficient state puts all five full Hessians below 5% without energy/force loss. Test100 and
validation remain untouched.

## HVP Curvature Funnel Update (2026-07-17)

Completed artifacts:

- train100 branch audit: `1200/1200` summaries under
  `/scratch/xzh/models/hvp_branch_stability/20260716/formal_train100_v2/train100_tasks`;
- single-seed A--E screen: 20 runs, 1200 optimizer steps each;
- three-seed comparison: 15 runs covering A, B and frozen C/D/E settings;
- Tier-1 summary:
  `/scratch/xzh/models/hvp_curvature_v1/20260716/multiseed_analysis/summary.json`;
- strict task table: 144 tasks over six runs, seven stable validation parents and 24 stable
  parent-directions at
  `/scratch/xzh/models/hvp_curvature_v1/20260716/strict_tasks_v1.tsv`.

Only C (`direct_hvp`, weight `1e-4`, floor `0.01`) passed the complete three-seed Tier-1 gate.
Relative to matched-seed A, its three-seed means improved validation energy MAE by 15.1%,
validation force MAE by 0.70%, and fixed-density HVP MAE by 0.86%; it won on 54.2% of directions
and 66.7% of eligible parents. These are validation-only fixed-density screening results, not a
strict complete-total or Test100 conclusion. Fixed-HVP relative Frobenius is effectively unchanged
(`1.04605` for A versus `1.04609` for C), so the candidate has not yet established broad curvature
or Hessian improvement.

The first multiseed evaluation submission failed before model loading because
`slurm_qm9_hvp_curvature_stage1_eval.sbatch` read a new eight-column TSV into seven shell
variables, appending `screen_selection_status` to `run_name`. The parser now consumes the optional
eighth field. The replacement array 2950 completed 15/15, analysis 2951 completed, and strict prep
2952 submitted strict array 2967. Existing checkpoints and labels were not modified or deleted.

Strict complete-total HVP array 2967 subsequently completed 144/144 tasks with zero failures, and
all six runs were numerically complete. C improved strict HVP MAE in all three seeds; its mean MAE
was `0.126903` versus `0.128389` for A (1.16% lower), with majority direction and parent wins in
two of three seeds. The preregistered strict gate passed. Analysis 2968 froze the median strict-HVP
seed-314159 C run for the full-Hessian check. Full-Hessian array 3112 contains five matched-A and
five C tasks over the same five validation molecules; at 12:09 it had completed 6/10 with four
long-tail tasks active on node01. It subsequently completed 10/10 with zero failures. C improved
the per-molecule mean relative full-Hessian MAE, RMSE and relative Frobenius by 2.73%, 2.83% and
2.83%, won Hessian MAE on 4/5 molecules, lowered mean frequency MAE by 2.14%, slightly improved
mode overlap, and reduced total imaginary-count error from 173 to 172. However, its mean pairwise
antisymmetric/symmetric Frobenius ratio worsened by 17.22%, above the frozen 5% limit. The full
Hessian/vibration gate therefore failed, `validation_promoted_candidates` is empty, and Test100
remains unread and unauthorized.

## Active Conservative total-OFDFT Derivative Work (2026-07-15)

The historical random1000 relaxed Hessians are still named
`incomplete-derived-force Hessian proxies`. A new implementation now assembles one scalar tensor
energy containing learned `kin_plus_xc`, Hartree, electron-nuclear, nuclear-nuclear, overlap/Pulay
and electron-number constraint terms. Do not merge the two result definitions.

New core files:

- `mldft/ofdft/geometry_integrals.py`
- `mldft/ofdft/conservative_force.py`
- `mldft/ofdft/stationary_density.py`
- `mldft/ofdft/implicit_response.py`
- `scripts/qm9_total_ofdft_force_audit.py`
- `scripts/qm9_total_ofdft_hvp_audit.py`
- `scripts/qm9_total_ofdft_hessian_audit.py`
- `docs/qm9_total_ofdft_conservative_force_hessian.md`

Additional changes provide tensor energies, analytic overlap derivatives, stable symmetric matrix
root VJPs, self-pair-safe nuclear repulsion and deterministic local-frame dummy positions. Focused
tests are under `tests/ofdft/`, `tests/ml/test_natural_reparametrization.py` and
`tests/ml/test_local_frames_module.py`.

Current implementation-audit evidence uses P1 EGF force-weight-1.0 s3000 and molecule `0000010`:

- every audited density is strict at about `6e-11` to `1e-10` projected gradient;
- tensor and legacy scalar total energies agree to about `1e-13 Ha`;
- complete total force coordinate 0 is `-68.14936925 Ha/Bohr`;
- its difference from independently relaxed scalar-energy FD decreases from `0.1120` at outer
  `h=3e-5` to `9.52e-5 Ha/Bohr` at `h=1e-6`;
- a model-geometry VJP step below `1e-6` enters cancellation; Richardson is slower and slightly
  worse, so `1e-6` remains the reference setting;
- two-dimensional complete-force loop work decreases
  `6.45e-4 -> 1.34e-7 -> 7.99e-9 Ha` for half-width
  `3e-4 -> 3e-5 -> 1e-5`;
- at HVP/response step `1e-5`, dense KKT density response agrees with strict relaxed density FD to
  `9.93e-5` relative Frobenius;
- the resulting complete implicit HVP agrees with strict relaxed total-force FD to `1.41e-4`
  relative Frobenius, and force/energy directional curvature agrees to the same scale;
- an independent coordinate-1 HVP agrees at `2.33e-7` relative Frobenius. The strict cross pair is
  `H10=-5211.2256`, `H01=-5209.9186 Ha/Bohr^2`, a `2.51e-4` relative mismatch; there is no evidence
  of structural nonconservativity at the measured numerical precision;
- the local total curvature is enormous (`-1.995e5 Ha/Bohr^2`, HVP norm `1.406e6`), so the old
  EGF model has a newly exposed physical second-order pathology even though the derivative
  implementation is internally consistent;
- the 393-dimensional tangent Hessian is symmetric positive definite but has condition number
  `7.60e6`. Dense reuse takes about 7.3 s; PCG/MINRES at 300-500 iterations is insufficient.
  Low-mode deflation works with exact modes but matrix-free ARPACK mode discovery is currently
  slower than dense assembly.

Superseding overlap/autograd and full-Hessian update:

- root cause of the analytic VJP mismatch was `AddOverlapMatrix` overwriting an already injected
  graph-connected overlap; existing overlap tensors are now preserved and regression-tested;
- pure transform-autograd total force differs from the numerical model-VJP force by only
  `7.35e-5 Ha/Bohr` and from relaxed scalar-energy FD by `1.69e-4 Ha/Bohr` at `h=1e-6`;
- pure-autograd `1e-5` loop work is `-7.76e-10 Ha`, with 4/4 strict corners;
- two strict HVP columns are reproduced by the full Hessian to `3.61e-10` and `3.44e-9` relative;
- `h=1e-3` and `3e-4` cross different strict density branches and are invalid for this model;
  `3e-5` and `1e-5` enter the same local branch and agree to `7.91e-3` in the symmetric Hessian;
- at `h=1e-5`, full total-Hessian asymmetry ratios are `9.41e-4` for EG and `1.17e-3` for EGF;
- physical model quality fails badly: relative Frobenius versus PBE is `1.51e6` for EG and
  `2.29e6` for EGF. The existing lambda=1 incomplete-force objective makes the true total Hessian
  about 51% worse on this molecule despite improving the historical proxy.

Persistent raw output:

```text
_runtime/qm9_p1_models/eval/qm9_total_ofdft_conservative_audit/20260715
```

Critical limitation: derivative conservativity is now locally validated, but the current EG/EGF
models were not trained against the complete total-energy force. Their stable local total Hessians
are physically pathological. Historical random1000/Test100 Hessians remain incomplete proxies.

Immediate continuation order:

1. repeat strict total force/HVP checks for EG and actual random1000 force-weight-1.0 across
   representative normal and hard molecules;
2. build the strict total-Hessian mini-benchmark using a molecule-specific step-stability gate;
3. train parent-grouped candidates that constrain complete total energy/force or conservative
   energy secants, then use the frozen
   three-tier evaluation funnel.

## Conservative Energy-Secant Candidate (2026-07-15)

The existing `ForceLoss` objective is not a complete-total-force objective: the model predicts
`kin_plus_xc`, while historical EGF training compares `-dE_model/dR` directly with total PBE
forces. The complete total-Hessian audit shows that this mismatch can improve the old proxy while
worsening the true relaxed total Hessian. The new candidate therefore constrains exact scalar
`kin_plus_xc` energy differences on train-parent-only `R-/R+` pairs instead of reinterpreting PBE
total forces as a learned-component derivative.

New training pieces:

- `PairEnergySecantLoss` in `mldft/ml/models/components/loss_function.py`;
- pair metadata loading in `mldft/ml/data/components/of_data.py`;
- `ParentPairBatchSampler` in `mldft/ml/data/components/pair_batch_sampler.py`;
- pair-aware loader/datamodule support in `mldft/ml/data/components/loader.py` and
  `mldft/ml/data/datamodule.py`;
- `configs/ml/model/loss_function/l1_energy_secant.yaml` with weights
  energy/gradient/secant `0.2/0.7/0.1` and no force loss;
- `configs/ml/experiment/str25/qm9_pbe_force_full_energy_secant.yaml`;
- one-line launchers `scripts/launch_qm9_random1000_energy_secant_8xa100.sh` and
  `scripts/slurm_qm9_random1000_energy_secant_8xa100.sbatch`.

Data and leakage contract:

- source: `QM9PBEForceRandom1000PairedAug`;
- 800 train parents, 1600 exact paired geometries;
- split sizes 4800/400/400 label files;
- parent overlap train/val/test is zero;
- split SHA256 is `919c5f6c250b0d2e9f14f43522894fcac8b179d6d4fd5b774c2cec1ea249034b`;
- only the final SCF sample from a paired geometry enters the secant; all non-final iterations
  from paired files are excluded from ordinary batch fill;
- complete pair batches are sharded across ranks and `set_epoch` is exposed to Lightning.

Validation status:

- focused local suite: 9 selected datamodule/sampler/secant tests pass;
- Slurm 800: single-A100 3-step smoke passed, finite energy/gradient/secant loss,
  `last.ckpt` saved, peak allocated CUDA memory 476 MiB;
- Slurm 802: real two-rank NCCL smoke passed (`MEMBER 1/2`, `2/2`), 3/3 updates and rank-zero
  artifacts; measured 15.11 global samples/s and 544 MiB peak allocated CUDA memory;
- Slurm 803: first 8-A100 attempt completed epoch 0 but failed during a duplicated validation
  pass. Disabling Lightning sampler replacement for the custom train sampler had also disabled
  validation sharding; 8 ranks x 8 workers each read 1251 batches/rank, aggregate RSS reached
  about 197 GB, and a shared-memory allocator cleanup raised SIGABRT. The model/loss path was
  finite; `epoch_000.ckpt` and `last.ckpt` were preserved;
- the datamodule now installs an explicit `DistributedSampler` for val/test/predict whenever pair
  training disables automatic replacement. With four workers/rank, validation is 157 batches/rank
  in about 10 s and aggregate RSS is about 110 GB;
- Slurm 804 resumed the exact run and completed epochs 1 and 2 with correctly sharded validation,
  but then hit the same PyTorch `file_system` `/torch_*` unlink SIGABRT at an epoch boundary. It
  preserved `epoch_001.ckpt`, `epoch_002.ckpt`, and `last.ckpt`;
- Slurm 822 resumed `epoch_002` in the same run directory with zero DataLoader workers. It has
  crossed multiple train/validation/checkpoint boundaries without the allocator failure and has
  preserved through `epoch_005.ckpt` at this status snapshot. Validation total/energy/gradient
  losses progressed `0.0184/0.0404/0.0129` after epoch 3,
  `0.0162/0.0304/0.0126` after epoch 4, and `0.0139/0.0207/0.0122` after epoch 5. The run directory is
  `/scratch/xzh/models/train/runs/qm9_random1000_energy_secant_w0p1_e10_20260715_171103`;
- zero workers reduced aggregate job MaxRSS from about 110 GB to 14.9 GB, but shared-storage/data
  transform throughput fell to roughly 1.1-1.3 steps/s. `mldft/ml/train.py` and the 8-A100
  launchers now expose `multiprocessing_sharing_strategy`, so `file_descriptor` with a small worker
  count can be calibrated separately without changing this recovery run.

The differentiable overlap contract is now explicit. `conservative_force.py` adds a one-shot
`preserve_injected_overlap_matrix` marker to its graph-connected overlap; `AddOverlapMatrix`
consumes that marker and otherwise retains the historical rebuild behavior. This preserves the
moving-basis/Pulay graph without reusing an already transformed overlap during inverse basis
reconstruction. Ten focused transform/conservative tests pass, including both master-transform
round trips.

Pending dependent evaluations:

- Slurm 872, node05: validation/test energy and historical incomplete-force regression plus
  10-molecule fixed-density learned-energy Hessian/HVP proxy;
- Slurm 873, node06: complete scalar-derived total-OFDFT KKT HVP versus PBE analytic `Hv` on
  `0000777`, `0040728`, `0003027`, followed by strict full total Hessian on `0000777`;
- both jobs depend on successful completion of Slurm 822 and compare against the actual
  force-weight-1.0 baseline checkpoint.

No model-improvement conclusion is available until 822, 872 and 873 complete. The fast force and
fixed-density outputs from 872 remain explicitly named incomplete learned-energy proxies. Only the
strict scalar-total-energy/KKT outputs from 873 can support a complete total-OFDFT curvature claim.

## Completed Random1000 Hessian Diagnosis (2026-07-14 to 2026-07-15)

This objective identified whether density-relaxed derived-force Hessian error and asymmetry come
from density residual, finite-difference displacement, numerical precision, or the incomplete
nuclear-force definition, then trained and independently evaluated improved candidates. The final
stable random1000 candidate is actual force weight 1.0; conclusions remain limited to this pilot.

Current force/energy definition audit:

- the learned target is `kin_plus_xc`;
- reported model force is only `-dE_model/dR` at fixed coefficients;
- density optimization minimizes a numerical total energy containing learned `kin_plus_xc`,
  Hartree, electron-nuclear attraction, and nuclear repulsion;
- PySCF-built classical integrals and nuclear repulsion are not differentiated with respect to
  `sample.pos` in the reported force;
- the response of the optimized density is also absent from the partial model-energy force;
- therefore the existing density-relaxed derived-force field is not guaranteed to be the gradient
  of the recorded relaxed total energy.

Representative set:

```text
0000777  natoms=7   EGF MAE outlier
0043905  natoms=14  central-error sample
0056566  natoms=16  EG median-error sample
0072895  natoms=17  EGF median-error sample
0054659  natoms=18  central-error sample
0093887  natoms=19  elevated relaxed-force asymmetry
0040728  natoms=21  dominant EG error/asymmetry outlier
0060531  natoms=27  largest-size representative
```

Active remote jobs and artifacts:

- `job 487`, node04, completed all 8 new `(h,tolerance)` conditions in 7:09:21 with zero worker
  failures; output:
  `/scratch/xzh/models/eval/qm9_random1000_hessian_protocol_scan/20260714_194632`;
- `job 505`, node05, completed closed-loop and relaxed scalar-energy 2x2 Hessian-block audits;
  output root:
  `/scratch/xzh/models/eval/qm9_random1000_hessian_physics_audits/20260714_213743`;
- dependency-waiting job 488 was cancelled before execution and replaced by job 505 on the idle
  node05; the scientific configuration is unchanged;
- `job 489`, node05, completed fixed-density `float32/float64` and three-displacement
  autograd/FD/HVP scan; output:
  `/scratch/xzh/models/eval/qm9_random1000_fixed_hessian_precision_scan/20260714_200631`.
- `job 492`, node01, controlled actual force-weight sweep `0.3,1,3,10`; job output root:
  `/scratch/xzh/models/train_lambda_sweep/20260714_202203`;
- `job 493`, node02, completed 1600 exact paired train-parent PBE labels in 9:11:22. Counts are
  1600 chk, 1600 finite-force labels, 800 complete parent pairs, zero failures;
- `job 494` failed during six-way force evaluation because 48 aggregate DataLoader workers caused
  a multiprocessing file-descriptor/shared-memory transfer failure. No model metric was used.
  Replacement `job 517`, node05, set `num_workers=0` and completed all six validation force plus
  fixed-density Hessian/HVP evaluations in 11:36; output:
  `/scratch/xzh/models/eval/qm9_random1000_lambda_tier1/20260714_234814`;
- `job 495` exposed a Hydra insertion error before transform. Replacements 525/532 completed the
  cached transform and strengthened leakage/path checks. The final augmented split has
  4800/400/400 train/val/test entries, 4000 base plus 1600 paired source labels, and zero parent
  overlap. Jobs 529 and 533 were preflight-only failures for an invalid unified-label-directory
  check and missing Hydra `+` on `limit_train_batches`; neither started model training. Job 536
  completed the 8-A100 paired actual-weight-3.0 compute-matched run in 49:18. It saved final
  `epoch_009.ckpt`, completed 12330 updates at mean 162.9 samples/s, and used split SHA256
  `919c5f6c250b0d2e9f14f43522894fcac8b179d6d4fd5b774c2cec1ea249034b`;
- `job 508` was cancelled before execution after its required `h=1e-3, tol=1e-5` source condition
  completed. Replacement 522 exposed a missing-final-analysis path and exited in one second.
  Replacement `job 523`, node06, read the explicitly generated
  `analysis_pre_rescue_20260715/per_displacement_optimization.csv`. Serial timing showed it would
  exceed eight hours, so 523 was cancelled after 43 minutes with its log retained. Replacement
  `job 524` completed the same 32 bad points over eight GPUs in 1:42:40. Its merger verified
  96/96 unique variant rows and 96/96 compressed cycle curves; output:
  `/scratch/xzh/models/eval/qm9_random1000_hessian_protocol_bad_point_rescue_parallel/20260715_031203`;
- `job 510`, node02, completed validation-set PBE analytic Hessians: 8/8 success in 3:27 wall;
  output: `/scratch/xzh/models/eval/qm9_random1000_validation8_pbe_hessian/20260714_221420`;
- `job 537`, node04, completed strict Val8 density-relaxed candidate comparison: 858/858 displaced
  points strict for every model; actual weights 3.0 and 1.0 were frozen for Test100, while the
  paired candidate failed the energy gate;
- `job 538`, node05, completed one-shot frozen Test100 in 2:45:38 with exit code zero. Actual
  weight 1.0 is the sole stable candidate under the frozen energy/force/strict-Hessian gates;
  output: `/scratch/xzh/models/eval/qm9_random1000_frozen_test100/20260715_081030`;
- `job 515`, node06, completed the six-GPU full-Hessian anomaly supplement at density threshold
  `1e-6` for `0000777` and `0040728` over all three requested steps in 7:25:04. Job 516 completed
  the unified raw/symmetrized, quantile, residual-correlation, and stability analysis in 2:38;
- `job 503`, node06, completed optional shared prepared-geometry cache correctness/timing smoke on
  `0000777` for both EG and historical EGF; output:
  `/scratch/xzh/models/eval/qm9_random1000_prepared_geometry_cache_smoke/20260714_job503`.
- `job 504`, node06, completed cache audit on `0040728`: Hessian relative differences below
  `5e-7`, 1.205x combined model-time speedup, 7.70 GB peak RSS; output:
  `/scratch/xzh/models/eval/qm9_random1000_prepared_geometry_cache_smoke/20260714_job504_0040728`;
- `job 507`, node06, completed final-density validation-only force and
  HVP-vs-PBE-force-secant runtime smoke before tier-1 job 494 starts; output:
  `/scratch/xzh/models/eval/qm9_random1000_validation_funnel_smoke/20260714_job507_ground_state`;
- job 506 exercised the older all-SCF compatibility path and is not used for model selection;
- `job 513`, node06, completed same-GPU concurrency calibration: four-molecule wall time
  `2640 -> 1395 s` (`1.892x`), 4/4 success, maximum Hessian relative difference `2.89e-5`;
- `job 514`, node06, completed CPU-thread calibration on `0040728`: 8 threads gave `1.203x`
  versus one thread, retained 126/126 strict points, and changed the Hessian by only `7.29e-7`
  relative Frobenius. Frozen Test100 therefore uses 16 shards, two workers per GPU, and eight
  CPU threads per worker;

Current quantitative evidence:

- on the selected 8 molecules at `h=1e-3`, tolerance `1e-4`, mean
  `||H_asym||F/||H_sym||F` is `0.200` for EG and `0.185` for the historical EGF model;
- symmetrization changes mean MAE only `0.0920 -> 0.0888` for EG and
  `0.01520 -> 0.01465` for EGF, so postprocessing does not explain or fix most PBE error;
- all baseline displacement points met `1e-4`; the completed loop audit shows density residual is
  not the leading smooth-case source;
- fixed-density full Hessians were finite in all 96 model/dtype/displacement cases;
- fixed-density autograd `float32-vs-float64` relative differences were approximately
  `9.8e-6` for EG and `2.9e-6` for EGF;
- fixed-density FD is most consistent near `h=1e-3`; `h=3e-4` shows more rounding noise, but
  this noise remains orders of magnitude below the relaxed-force asymmetry;
- on `0000777`, tightening `1e-4 -> 1e-5 -> 1e-6` changes EG curl only
  `-0.107629 -> -0.107593 -> -0.107581` and EGF curl only
  `-0.059058 -> -0.058963 -> -0.058985`; each is 99.8%--100.0% of the corresponding baseline
  maximum Hessian antisymmetry;
- the same curl is stable at `h=3e-3,1e-3,3e-4`, while fixed-density loop work remains near zero.
  Pearson correlation of final density residual with absolute curl is only 0.109 (EG) and 0.148
  (EGF). This assigns the dominant smooth-case asymmetry to the incomplete force definition;
- naturally symmetric relaxed-total-energy 2x2 blocks are step stable on strict points. On common
  strict molecule `0021889`, EG total-energy block changes only 0.88% relative Fro between steps,
  but current force-vs-total-energy MAE is 1.5845. Historical EGF is 9/9 strict for all three
  audited molecules at both steps, with 0.22%--0.86% total-energy block changes and mean
  force-vs-total-energy MAE about 0.569;
- `0040728` remains a numerical/branch hard case: curl is step dependent and several `1e-6`
  corners fail. It is reported separately rather than used to infer a smooth local Hessian;
- Test100 symmetrized vibrational diagnostics give mean frequency MAE `1779.9 cm^-1` (EG) and
  `575.2 cm^-1` (EGF), but mean mode overlap is `0.636` and `0.618`, and both models produce
  about 1260 imaginary modes versus 229 in the PBE references. Frequency claims are not yet
  physically satisfactory.
- the complete 3x3 scan has 834/834 strict points for both models at every `1e-4` condition.
  At `1e-5`, only 806--824/834 points are strict because many runs hit the 10000-cycle cap;
  tighter requested tolerance is not tighter realized convergence;
- the five regular representatives change by at most about `2.8e-3` relative Frobenius across
  steps, while `0040728`, `0060531`, and `0093887` change by `0.14`--`1.38`. This fixes
  `h=1e-3, tol=1e-4` as the formal proxy compromise and keeps hard cases separately flagged;
- the full `1e-6` anomaly supplement confirms that EGF `0000777` maximum antisymmetry remains
  `0.05972/0.05899/0.05900` across the three steps, matching the closed-loop curl. In contrast,
  `0040728` reaches only 67--76/126 strict EGF displacement points and its maximum antisymmetry
  changes `0.461 -> 1.244 -> 3.616` as the step shrinks; it is a solver/branch hard case, not a
  trusted local tensor;
- on the 32 points that missed `1e-5`, label warm-start plus Adam converged only 7/18 EG and
  8/14 EGF points; adding SLSQP improved this to 15/18 and 10/14. SAD plus Adam-to-SLSQP reached
  18/18 and 14/14 with maximum final gradients below `1e-5`. The strict rescue policy is therefore
  cheap base-density two-stage Adam first, followed only for failures by SAD + Adam-to-SLSQP;

Training configuration correction:

- the historical random1000 checkpoint named `EGF_lam1` actually has
  `model.loss_function.force_loss.weight=0.1` in `hparams.yaml`;
- P1-410 `lambda=1.0` used an actual force weight of `1.0`;
- treat the random1000 historical model as nominal-lambda1/effective-weight-0.1;
- the prepared sweep trains actual force weights `0.3, 1.0, 3.0, 10.0` with the same seed;
- do not compare models by the old run name alone.

Validation-only Tier-1 weight selection is complete in replacement job 517:

- all 120 full fixed-density Hessians and 480 HVP cases were finite;
- actual weight 3.0 is first: energy MAE `0.022768`, force-component MAE `0.002292`,
  HVP-vs-PBE-force-secant MAE `0.018609`, relative Frobenius `0.3784`;
- actual weight 1.0 is second: `0.023097`, `0.002496`, `0.021242`, `0.4342` respectively;
- historical effective weight 0.1 gives `0.021261`, `0.003624`, `0.025506`, `0.5186`;
- weight 10 has the best force/HVP values but energy MAE `0.024450` exceeds the pre-registered
  eligibility limit `0.0233866`, so it is not advanced;
- the train-parent paired candidate therefore uses actual force weight 3.0. Test100 was not read
  for this decision.

Validation-only Tier-2 freezing is complete in job 537:

- every model reached strict density convergence at all 858 displaced points;
- actual weight 3.0: energy/force/Hessian MAE `0.022767/0.002292/0.012200`, relative Frobenius
  `0.5075`; frozen for Test100;
- actual weight 1.0: `0.023099/0.002496/0.013372`, relative Frobenius `0.5959`; frozen for Test100;
- paired actual-weight-3.0: `0.024181/0.002275/0.012128`, relative Frobenius `0.5004`; its small
  force/Hessian gain did not compensate for failing the `0.0233691` energy gate, so it was not
  exposed to Test100;
- historical weight 0.1 and EG remain controls. The frozen artifact is
  `/scratch/xzh/models/eval/qm9_random1000_validation_tier2_strict_hessian/20260715_064755`.

`qm9_density_relaxed_tier2_analysis.py` and `qm9_test100_funnel_analysis.py` now derive convergence
aggregates from per-point optimization rows when the evaluator summary omits them; focused tests in
`tests/test_qm9_funnel_analysis.py` protect this output contract.

Frozen Test100 confirmation is complete in job 538:

- all 400 fixed full autograd Hessians and 1600 HVP cases are finite;
- both new candidates completed 100/100 Hessians and 10638/10638 strict displacement points;
- actual weight 1.0: energy/force/fixed-H/relaxed-H MAE
  `0.027573/0.002463/0.009316/0.010291`; relative to historical weight 0.1 these are
  `+4.79%/-31.16%/-13.12%/-14.77%`. It is the stable candidate;
- actual weight 3.0 has stronger force/relaxed-H gains but energy ratio `1.10573` exceeds the
  frozen `1.10` limit, so it remains a Pareto candidate only;
- weight 1.0 improves relaxed-Hessian MAE on 95/100 molecules. Fixed and relaxed evaluation choose
  the same best model on 86/100, supporting fixed autograd/HVP as screening but not replacement;
- current relaxed result remains an incomplete-derived-force proxy, not a conservative total-OFDFT
  Hessian. Full analysis is in
  `docs/qm9_random1000_density_relaxed_hessian_diagnosis_and_optimization.md`.

Paired-geometry preparation:

- `QM9GeometryPerturbed` now supports exact deterministic `+delta/-delta` pairs while preserving
  the old independent-perturbation default;
- the existing random1000 split is parent-grouped and has 800/100/100 train/val/test parents with
  zero overlap;
- a train-only raw subset with 800 parent symlinks is prepared at
  `/scratch/xzh/data/QM9PBEForceRandom1000PairedTrain/raw`;
- manifest SHA256 for the sorted train-parent IDs is
  `c3ffc35b2b930caefae9ac598e8a8d03e21fba73f612443b83d9e2509b5a8e30`;
- job 493 completed 1600 paired PBE labels for those 800 train parents only; no val/test parent is
  linked into the source directory;
- the completed augmented split preserves original validation/test entries and statistics, adds
  paired entries only to train, and has zero parent overlap. Its compute-matched weight-3 candidate
  was rejected by the independent Val8 energy gate before Test100.
- Tier 1 uses validation ground-state energy/force and fixed-density HVP against validation PBE
  force secants. Tier 2 uses the new independent Val8 PBE analytic Hessians. Historical Test100
  results were used for protocol diagnosis, but no new candidate sees its Test100 result until the
  candidate list is frozen by Tier 2.

Second-order smoothness audit:

- output: `/scratch/xzh/models/eval/qm9_random1000_model_second_order_smoothness/20260714`;
- actual activations are GELU and SiLU; values and first/second derivatives were finite in the
  float64 sweep;
- node and attention cutoffs are `null`;
- all eight original representative structures have a fixed full directed graph with exactly
  `N^2` unique edges and `N` constant-distance self loops;
- activation, cutoff, neighbor switching, and float precision are not leading sources in the
  audited fixed-density path;
- geometry-dependent cached local-frame transforms remain outside the current nuclear autograd
  path across independently rebuilt displaced samples and are part of the total-derivative risk.

Density-relaxed cost work:

- Test100 model-time accounting assigns about 72.1% (EG) and 80.4% (historical EGF) to density
  optimization; sample/integral construction and force evaluation are the remaining material cost;
- pair-response extrapolation reduced `0000777` minus-point mean cycles by 19.5%, but changed
  `0040728` by only 0.3%; it is an opt-in heuristic, not a general default;
- extrapolated and baseline Hessians differ by relative Frobenius `7.0e-4` on `0000777` and
  `2.64e-4` on `0040728`, with practically unchanged PBE metrics;
- `qm9_hessian_density_relaxed_eval.py` now records split sample-build, density-optimization,
  force-autograd, and total-point timings;
- optional `--share-prepared-geometry-across-runs` stores prepared model-independent geometry
  samples on CPU for EG/EGF reuse. Job 503 obtained 43 builds/43 hits, cached-vs-uncached relative
  Frobenius differences `7.4e-9` (EG) and `8.74e-5` (EGF), and only 1.022x combined model-time
  speedup. Trial it on the representative tier before Test100. Jobs 499/500 were invalid wrappers;
  duplicate-output jobs 501/502 were cancelled. None produced usable scientific results.
- same-GPU concurrency and CPU-thread calibration support the frozen 16-shard/two-worker-per-GPU,
  eight-thread-per-worker configuration. The measured speedups are `1.892x` and `1.203x`, with
  all audited points strict and negligible Hessian changes;

New or materially changed scripts:

- `scripts/qm9_hessian_density_relaxed_eval.py`;
- `scripts/qm9_hessian_protocol_scan_analysis.py`;
- `scripts/qm9_force_conservativity_loop_audit.py`;
- `scripts/qm9_relaxed_scalar_energy_hessian_audit.py`;
- `scripts/qm9_hessian_vibrational_metrics.py`;
- `scripts/qm9_second_order_autograd_hessian_audit.py`;
- `scripts/launch_qm9_random1000_hessian_protocol_scan_node04.sh`;
- `scripts/launch_qm9_random1000_hessian_physics_audits_node04.sh`;
- `scripts/launch_qm9_random1000_fixed_hessian_precision_scan_node05.sh`;
- `scripts/prepare_qm9_paired_train_subset.py`;
- `scripts/launch_qm9_random1000_force_lambda_sweep_8xa100.sh`.
- `scripts/qm9_fixed_hessian_precision_scan_analysis.py`;
- `scripts/qm9_model_second_order_smoothness_audit.py`;
- `scripts/qm9_hessian_physics_audit_analysis.py`;
- `scripts/launch_qm9_random1000_paired_train_labels.sh`;
- `scripts/prepare_qm9_paired_augmented_training.py`;
- `scripts/launch_qm9_random1000_paired_augmentation_prepare.sh`;
- `scripts/launch_qm9_random1000_lambda_tier1_screen.sh`;
- `scripts/qm9_lambda_sweep_tier1_analysis.py`.
- `scripts/launch_qm9_prepared_geometry_cache_smoke.sh`;
- `scripts/slurm_qm9_prepared_geometry_cache_smoke.sbatch`.
- `scripts/launch_qm9_random1000_paired_candidate_8xa100.sh`;
- `scripts/slurm_qm9_random1000_paired_candidate_8xa100.sbatch`.
- `scripts/qm9_prepared_geometry_cache_analysis.py`.
- `scripts/select_qm9_split_representatives.py`;
- `scripts/launch_qm9_validation_funnel_smoke.sh`;
- `scripts/slurm_qm9_validation_funnel_smoke.sbatch`.
- `scripts/launch_qm9_random1000_protocol_bad_point_rescue.sh`;
- `scripts/launch_qm9_random1000_validation8_pbe_hessian.sh`;
- `scripts/launch_qm9_random1000_validation_tier2_strict_hessian.sh`;
- `scripts/qm9_density_relaxed_tier2_analysis.py`;
- `scripts/launch_qm9_random1000_frozen_test100.sh`;
- `scripts/qm9_test100_funnel_analysis.py`.
- `scripts/launch_qm9_random1000_same_gpu_concurrency_calibration.sh`;
- `scripts/launch_qm9_random1000_cpu_thread_calibration.sh`;
- `scripts/launch_qm9_random1000_anomaly_tol1e6_full_hessian.sh`;
- `scripts/slurm_qm9_random1000_anomaly_tol1e6_full_hessian_node06.sbatch`.
- `scripts/slurm_qm9_random1000_protocol_combined_analysis.sbatch`.
- `scripts/qm9_bad_point_rescue_merge.py`;
- `scripts/launch_qm9_random1000_protocol_bad_point_rescue_parallel.sh`;
- `scripts/slurm_qm9_random1000_protocol_bad_point_rescue_parallel_node05.sbatch`.
- `scripts/legacy_dft_model_smoke.py`.

## Legacy `/scratch/xzh/dft` Model Recovery (2026-07-15)

The older perturbed-Fock data and Graphformer checkpoints staged under `/scratch/xzh/dft` are
restored and independently smoke-tested on node01. They are SCF-trajectory/density-gradient data,
not the current force/Hessian label datasets.

- the remote QM9 and QMUGS checkpoint SHA256 values exactly match the local
  `_runtime/models/train/runs/trained-on-qm9` and `trained-on-qmugs` copies;
- installed `tensorframes==1.0.0` from the project-locked Git commit
  `cd1addfd3c82a47095c9961ab999dcabfab4c21d`; the transferred wheel is retained at
  `/scratch/xzh/dft/restore/wheels/tensorframes-1.0.0-py3-none-any.whl` with SHA256
  `674fd49d18f97cdf90c525f49b2c7f58c0d1c5d3b0c8204d60bc02c5930e3a39`;
- recovered `dataset_info.yaml` records `6-31G(2df,p)` plus `even_tempered_2.5` in
  `QM9_perturbed_fock`, `QMUGSBin0_perturbed_fock`, and
  `QMUGSBin0QM9_perturbed_fock`. The reconstructed CH4 density-basis dimension is 189, exactly
  matching archived QM9 label `0000001`;
- transformed label caches are not staged, so the smoke script sets `use_cached_data=false` and
  reconstructs the configured local-frame/global-natural-representation transforms from raw
  `labels`;
- both checkpoint states contain zero non-finite values. The QM9 test-split smoke produced finite
  energy/gradient/difference outputs for a 6-atom, 305-coefficient sample; the QMUGS train-split
  smoke passed for a 26-atom, 1602-coefficient sample;
- reproducible remote script and report:
  `/scratch/xzh/dft/restore/legacy_dft_model_smoke.py` and
  `/scratch/xzh/dft/restore/smoke_report_20260715.json` (`status: passed`);
- use raw `hparams.yaml` with `DFT_DATA=/scratch/xzh/dft/data` and
  `DFT_MODELS=/scratch/xzh/dft/models`. The preserved `hparams_resolved.yaml` files contain stale
  `/export/scratch/ialgroup/...` provenance paths and must not be used as current paths.

### Full recovery inventory and content audit (2026-07-15)

The single-sample smoke has now been followed by a read-only full archive audit. The authoritative
remote reports are:

- `/scratch/xzh/dft/restore/recovery_manifest_20260715.json`;
- `/scratch/xzh/dft/restore/content_audit_full_20260715/report.json`;
- `/scratch/xzh/dft/restore/split_leakage_audit_with_metadata_20260715.json`;
- `/scratch/xzh/dft/restore/group_safe_splits_20260715/report.json`.

All 158,455 physical label archives were opened and every numeric Zarr array was read. The audit
covered about 247.9 GB and 5,446,247 stored SCF configurations:

| source | archives | SCF steps | bytes | schema/nonfinite/shape failures |
| --- | ---: | ---: | ---: | ---: |
| `QM9_perturbed_fock` | 133,884 | 4,693,856 | 198,779,641,309 | 0 |
| `QMUGS_perturbed_fock` | 23,721 | 742,952 | 46,635,464,852 | 0 |
| `QMUGS` | 850 | 9,439 | 2,461,730,725 | 0 |

The maximum labeled-step electron-number relative errors are `5.39e-4`, `1.23e-4`, and
`6.32e-5`, respectively, all below the historical label generator warning threshold `1e-3`.
Exact rounded geometry, distance-invariant geometry, whole trajectory, and every individual
geometry+density-configuration SHA256 have zero cross-split duplicates in all three virtual
datasets. The compressed anomaly and duplicate lists are empty.

The archived split files are label-, conformer-, and parent-disjoint, but they are not fully
structure-group-disjoint:

- QM9 has 32 canonical-SMILES values crossing train/validation/test;
- QMUGS Bin0 has 50 crossing train/validation;
- the archived combined split has 66 crossing train/validation;
- the archived combined split also moves all 13,388 historical QM9 validation labels into train
  and all 13,389 historical QM9 test labels into validation. It must not be used for independent
  model selection or final testing.

Original split files were not modified. Sidecar group-safe split files were generated below
`/scratch/xzh/dft/restore/group_safe_splits_20260715`; canonical-SMILES and parent overlap is zero
by construction and by a separate postcondition check. Their pickle SHA256 values are:

- QM9: `58d012cee4f3f1817971081f6267a96591812397584636ce4d8a0c4e85eb6b99`;
- QMUGS Bin0: `3f54320dc475c4e609e3e66a27e936547ac425366f3f952e2379c394cbd17ab2`;
- combined: `c3e7fdbe9ddadc40d916a941a0db68900d2d17a6c00ba7d7f3ee7c8c7d48c0a8`.

The combined sidecar split reconstructs the source-domain QM9/QMUGS assignment before grouping;
its train/validation/test molecule counts are 128,266/15,934/14,255. This is an identity-only
repair, never a metric- or test-performance-based selection.

Effective paths are guarded in every new recovery/evaluation entry point. The negative test at
`/scratch/xzh/dft/restore/path_guard_negative_20260715.json` deliberately passes an
`/export/scratch/ialgroup/...` data root and confirms immediate nonzero exit. Archived stale paths
remain recorded only as ignored provenance.

### Historical-model recovery acceptance and derivative boundary (2026-07-15)

The full recovery objective is broader than the earlier one-sample smoke. Its authoritative
machine artifacts live under `/scratch/xzh/dft/restore`; the final aggregate is
`final_acceptance_report_20260715.json` with a human-readable companion
`final_acceptance_report_20260715.md`. Neither source data nor the two original checkpoints is
modified by this work.

Frozen environment and immutable model identity:

- Python `3.11.15`, torch `2.4.1`, torch-geometric `2.6.1`, Lightning `2.5.0`, Hydra `1.3.4`,
  PySCF `2.13.1`, Zarr `2.18.4`, NumPy `2.4.6`, and tensorframes `1.0.0` are recorded in
  `/scratch/xzh/dft/restore/recovery_manifest_20260715.json`;
- QM9 checkpoint SHA256 remains
  `9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09`;
- QMUGS checkpoint SHA256 remains
  `dde9e2e940ebbfcf4c74681b3264c1add71bf3539634e1b81bacffd5bd08be32`;
- all three `dataset_info.yaml` files are byte-identical, SHA256
  `9dfa58a92bb5c578f3a96d6562fd0f9ed0ee6b18b023b7a4f575239b84badb56`;
- raw `hparams.yaml` is the executable configuration. Stale paths in
  `hparams_resolved.yaml` are preserved only as provenance; every recovery entry point rejects an
  effective `/export/scratch/ialgroup` path before data or model evaluation.

The recovery acceptance is frozen in
`/scratch/xzh/dft/restore/model_acceptance_20260715/report.json`. For each historical model it
uses seven representative molecules spanning train/validation/test, atom-count, element,
coefficient-dimension, and source extremes, with initial, intermediate, and final density points
(21 samples per model). It checks repeated CPU float64, GPU float32, GPU float64, batch size greater
than one, coefficient ordering, basis metadata, electron constraints, and finite energy,
density-gradient, and density-difference outputs. Maximum GPU-float32 deviations from CPU-float64
are:

| model | energy max abs/electron | energy relative L2 | density-gradient max abs | density-gradient relative L2 |
| --- | ---: | ---: | ---: | ---: |
| QM9 | `2.201e-6` | `5.419e-7` | `3.513e-6` | `4.871e-7` |
| QMUGS | `2.687e-6` | `5.008e-7` | `4.878e-6` | `6.468e-7` |

Both pass the preregistered extensive-energy gate of `5e-6` relative/per electron and the
coefficient-output gate of `5e-4`. GPU float64 differences are much smaller. Because no trusted
historical per-sample predictions were present, deterministic CPU-float64 outputs were frozen as
golden only after all device and batching checks passed. Their report SHA256 values are
`431aa8975adb484bd6f50bb92e63a1667f2789713f0bbadf64aa9ec4d984acb3` (QM9) and
`dcfbffc23a8c291c7eef5308066e1aaa41f74c039dcf52fd5cebf4cb36672efb` (QMUGS).

Scientific forward baselines use only the group-safe sidecar split. Each validation/test label is
evaluated at archived SCF step 6 and at its final density. This distinction is mandatory: the
final-density difference target is identically zero, while representative CPU-float64 perturbed
points have nonzero per-coefficient difference MAE (`0.0040--0.00635` for QM9 and
`0.00301--0.00583` for QMUGS). The final reports therefore group metrics by density role rather
than presenting the zero final target as a meaningful difference benchmark. Validation is frozen
before the one and only Test forward evaluation. Per-sample rows, grouped source/atom/composition
metrics, quantiles, top-100 outliers, command lines, costs, and SHA256 values are retained under:

- `/scratch/xzh/dft/restore/scientific_baseline_validation_full_20260715`;
- `/scratch/xzh/dft/restore/scientific_baseline_test_full_20260715`.

The frozen aggregate forward metrics are below. Each row combines the SCF-6 and final-density
evaluations, so the number of rows is twice the number of physical labels. Energy is in Hartree;
gradient and difference are pooled per-coefficient errors in the archived representation.

| partition | model/domain | rows | energy MAE/RMSE | projected-gradient MAE/RMSE | difference MAE/RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | QM9 model / QM9 | 26,804 | `0.00123362/0.00232658` | `0.0156113/0.0297833` | `0.00386084/0.00923176` |
| validation | QMUGS model / QMUGS | 5,064 | `0.00299506/0.00376861` | `0.0141301/0.0277270` | `0.00384173/0.00908314` |
| validation | QM9 model / combined | 31,868 | `0.00417433/0.00988675` | `0.0163466/0.0304387` | `0.00385666/0.00919946` |
| validation | QMUGS model / combined | 31,868 | `0.00366966/0.00470382` | `0.0148756/0.0283695` | `0.00385666/0.00919946` |
| Test | QM9 model / QM9 | 26,810 | `0.00121182/0.00204024` | `0.0156045/0.0297716` | `0.00386468/0.00923948` |
| Test | QMUGS model / QMUGS | 1,698 | `0.0245065/0.0349810` | `0.00515204/0.00990771` | `7.52458e-5/4.35533e-4` |
| Test | QM9 model / combined | 28,510 | `0.0274281/0.133212` | `0.0140624/0.0266509` | `0.00275688/0.00777604` |
| Test | QMUGS model / combined | 28,510 | `0.00507428/0.00981639` | `0.0123842/0.0249177` | `0.00275688/0.00777604` |

Density-role separation remains essential. For example, the QM9-on-QM9 Test energy/gradient/
difference MAEs are `0.000638328/0.00213469/0` at the final density and
`0.00178531/0.0290743/0.00772936` at SCF step 6. The QMUGS-on-QMUGS Test values are
`0.0244694/0.00502012/0` and `0.0245435/0.00528396/0.000150492`, respectively. On these
validation and Test densities both historical difference branches return zero; consequently the
same-domain/combined difference errors are the frozen zero-output baseline (the magnitude of the
nonzero target at SCF step 6), not evidence that the two learned branches are equally accurate
away from this evaluation set and not a merge defect.

The combined Test source groups expose the cross-domain boundary. The QM9 model has energy MAE
`0.00121180` on the QM9 rows but `0.440874` on the 1,700 QMUGS rows. The QMUGS model has energy
MAE `0.00384140` on QM9 and `0.0245176` on those QMUGS rows. These are archived-label forward
errors, not optimized-density errors.

The primary Test array used `batch_size=64`; its eight QM9-model/combined-domain tasks exhausted
one 80-GB A100 on the largest QMUGS graphs and produced no summaries or accepted row artifacts.
The OOM logs are retained under
`scientific_baseline_test_full_20260715/qm9_model__combined_domain/failed_batch64_job793` and are
excluded from science. Only those eight missing shards were rerun at `batch_size=1` with the same
frozen checkpoint, split, SCF selectors, float32 dtype, cache and metrics. All eight passed, all
four merges passed, and Slurm provenance records the exact failed and replacement task sets.

The final aggregate and requirement audit both pass. The authoritative machine report SHA256 is
`97ba90c732421e19d0fba04f211ebc32ce35364190cb98d2bb74e033c46abdf7`; the rendered report
SHA256 is `e8b0a44e0c65e4132120b28fa243996e3b9b0913b01eb64a64baf959c5ff1c29`.
The completion audit is rerun after synchronizing this handoff so its evidence records the final
handoff SHA rather than the pre-sync copy.
A repository copy of the complete report is
[`legacy_dft_recovery_acceptance_20260715.md`](legacy_dft_recovery_acceptance_20260715.md).

Validation-only representative density optimization is frozen at
`/scratch/xzh/dft/restore/density_optimization_validation_20260715/report.json`:

- QM9 is 3/3 strict, with mean/max cycles `381.7/510`, maximum projected-gradient residual
  `9.87e-5`, mean optimized-density L2 error `0.0515`, and maximum electron error `1.14e-13`;
- QMUGS is 2/3 strict. `0987810.zarr.zip` is a real scientific nonconvergence: stage 1 reached
  about `9.57e-3`, stage 2 exhausted 10000 cycles and ended at `6.16e-2` after about `876.5 s`;
- label-density forward energy error, optimized-density forward energy error, coefficient error,
  density L2 error, cycles, residuals, electron constraint and cost are separate fields. A launch
  or infrastructure failure is never counted as a scientific result.

Force/Hessian integration is audited in
`/scratch/xzh/dft/restore/force_hessian_boundary_report_20260715.json` with units Hartree, Bohr,
Hartree/Bohr and Hartree/Bohr²:

- validation fixed-density scalar-energy autograd produces finite full Hessian and HVP results at
  `h=1e-3 Bohr` and is compared with finite differences;
- for the separately trained, actual force-weight `1.0` EGF candidate, validation Tier 2 has
  energy/force/relaxed-proxy-Hessian MAE `0.023099/0.002496/0.013372` and 858/858 strict points;
- the frozen Test100 gives energy/force/fixed-Hessian/relaxed-proxy-Hessian MAE
  `0.027573/0.002463/0.009316/0.010291`; fixed autograd-vs-FD mean MAE is `2.11e-6`, all 100 full
  Hessians and 400 HVP cases are finite, and HVP-vs-PBE-force-secant MAE is `0.019276`;
- the relaxed proxy completed 100/100 Hessians and 10638/10638 strict points, but it is still a
  finite difference of an incomplete derived force.

Terminology is frozen: the recovered Graphformer checkpoints are energy/density-gradient/
difference models, not force/Hessian models. Fixed-density coordinate autograd omits density
response and the coordinate response of classical integrals/nuclear/Pulay terms. The relaxed
derived-force tensor is an incomplete proxy, not a conservative physical total-OFDFT Hessian.
No physical vibrational conclusion is permitted from either.

The conservative total-OFDFT implementation plan and graph-break audit are in
`total_derivative_remote_readiness_20260715.json` and
`total_derivative_local_readiness_20260715.json`. The remote recovery snapshot does not yet contain
the candidate tensor-energy, moving-integral/Pulay, conservative-force, or implicit-response
interfaces. The local candidate has 25/25 unit tests passing, covering tensor scalar energies,
nuclear repulsion finite differences, classical/Pulay bundles, the electron constraint, and
PCG/MINRES/KKT response. Deployment remains blocked on a clean code-release audit, not on a claim
of physical completion. The required staged gates are:

1. remove `.item()`, `float`, NumPy, detach and CPU graph cuts before the reporting boundary;
2. assemble learned, Hartree, electron-nuclear and nuclear-nuclear scalar energy as tensors;
3. add audited coordinate JVP/VJP for Coulomb, attraction, overlap/metric, dual integrals,
   transformations, moving basis/local frames, normalization and every Pulay term;
4. validate constrained envelope total force against completely reoptimized central differences
   and step-convergent zero closed-loop work;
5. solve tangent-space KKT density response and implicit Hessian-vector products, then check force
   finite differences, symmetry, translations and rotations;
6. freeze validation tolerances before one Test evaluation; authorize mass-weighted physical
   vibration only after every earlier gate passes.

Failed harness/protocol attempts are retained in
`/scratch/xzh/dft/restore/failure_inventory_20260715.json`, including the non-covering SQLite merge,
two batching harness defects, the invalid fixed extensive-energy absolute gate, redundant raw
transform attempt, missing remote derivative tests, and GPU transform smoke. Every entry is marked
`used_as_scientific_result=false` with cause, resolution, logs and hashes. Jobs 704/705 were
cancelled before execution when the final-density-only difference baseline was recognized as
scientifically trivial; no Test metric was observed from them.

Recovery-specific reproducible entry points added in `scripts/` include:

- `legacy_recovery_manifest.py`, `legacy_dataset_content_audit.py`,
  `legacy_merge_content_audit.py`, `legacy_split_leakage_audit.py`, and
  `legacy_build_group_safe_splits.py`;
- `legacy_model_acceptance.py`, `legacy_merge_model_acceptance.py`, and
  `legacy_density_optimization_eval.py`;
- `legacy_build_baseline_cache_manifest.py`, `legacy_prepare_baseline_cache_shard.py`,
  `legacy_merge_baseline_cache.py`, `legacy_scientific_baseline_shard.py`, and
  `legacy_merge_scientific_baseline.py`;
- the validation/Test cache, forward, merge and freeze Slurm wrappers, including the explicit
  SCF-6 cache arrays;
- `legacy_force_hessian_artifact_audit.py`,
  `legacy_total_derivative_readiness_audit.py`, `legacy_immutability_audit.py`,
  `legacy_freeze_slurm_wrappers.py`, `legacy_failure_inventory.py`,
  `legacy_pipeline_provenance.py`, `legacy_final_acceptance_report.py`,
  `legacy_render_final_acceptance_report.py`, and `legacy_completion_audit.py`.

The authoritative report embeds representative commands, while every shard summary retains its
exact `sys.executable + sys.argv`, device, dtype, batch size, edge policy, host, input hashes,
elapsed time and output SHA256.

## Workspace

Repository:

```bash
/mnt/afs/home/xiazhenhao/dft/structures25
```

Important runtime roots:

```bash
_runtime/qm9_p1
_runtime/qm9_p1_models
```

Typical environment:

```bash
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models
```

Remote 8-GPU environment:

```bash
ssh -J root@101.200.216.125 -p 2200 shenwei01@localhost
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
```

Remote roots:

```bash
/scratch/xzh/code/structures25
/scratch/xzh/data
/scratch/xzh/models
```

## Main Code Entry Points

Training and force evaluation:

- `scripts/append_gpu4pyscf_force_labels.py`
- `scripts/compare_force_label_dirs.py`
- `scripts/launch_gpu4pyscf_force_labels.sh`
- `scripts/qm9_force_eval.py`
- `scripts/qm9_force_forward_backward_smoke.py`
- `scripts/qm9_force_mldft_smoke.py`
- `scripts/launch_qm9_full_scale_8xa100.sh`
- `scripts/launch_qm9_full_scale_4gpu.sh`
- `scripts/launch_qm9_throughput_calibration.sh`
- `scripts/launch_qm9_random1000_train_8xa100.sh`
- `scripts/slurm_qm9_random1000_train_8xa100.sbatch`
- `configs/ml/experiment/str25/qm9_pbe_force_pilot_eg.yaml`
- `configs/ml/experiment/str25/qm9_pbe_force_pilot_egf.yaml`
- `configs/ml/experiment/str25/qm9_pbe_force_full_eg.yaml`
- `configs/ml/experiment/str25/qm9_pbe_force_full_egf.yaml`
- `configs/ml/callbacks/qm9_full_scale.yaml`
- `configs/ml/callbacks/throughput_monitor.yaml`
- `configs/ml/trainer/ddp_8xa100.yaml`
- `mldft/ml/callbacks/throughput.py`
- `mldft/utils/log_utils/hydra_callbacks.py`

Hessian and density-relaxed evaluation:

- `scripts/qm9_hessian_eval_reference_set.py`
- `scripts/qm9_hessian_reference_eval.py`
- `scripts/qm9_hessian_density_relaxed_eval.py`
- `scripts/qm9_hessian_directional_eval.py`
- `scripts/qm9_pbe_hessian_reference_set.py`
- `scripts/qm9_mass_weighted_hessian_check.py`
- `scripts/launch_qm9_gpu4pyscf_hessian_validation.sh`
- `scripts/launch_qm9_random1000_test100_hessian_eval_node04_gpu4pyscf.sh`
- `scripts/slurm_qm9_random1000_test100_hessian_node04_gpu4pyscf.sbatch`
- `scripts/launch_qm9_random1000_test100_density_relaxed_hessian_node04.sh`
- `scripts/slurm_qm9_random1000_test100_density_relaxed_hessian_node04.sbatch`

Second-order autograd:

- `scripts/qm9_second_order_autograd_hessian_audit.py`
- `mldft/ml/models/components/gbf_module.py`
- `tests/ml/test_gbf_module.py`
- `tests/ml/test_qm9_second_order_autograd_runtime.py`

Recent critical model fix:

- `_safe_edge_lengths()` in `mldft/ml/models/components/gbf_module.py` makes self-loop edge distances second-order-safe.
- For `i == j`, distance is constant zero and does not call `torch.norm(0)`.
- For `i != j`, distance still uses the original `torch.norm(pos_i - pos_j)`.

## Data and Checkpoints

Validated P1-410 snapshot:

- 410 molecules;
- 1640 labels;
- `.chk` and labels checked;
- force labels checked finite, no NaN/Inf in the validated snapshot.

Main trained checkpoints:

| model | checkpoint |
| --- | --- |
| EG_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt` |
| EGF_lam1_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt` |

Main PBE Hessian references:

- 20-molecule reference manifest: `_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json`
- PBE Hessian cache: `_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessians/`

## Current Results

### Training

Completed:

- EG baseline trained to s3000.
- EGF lambda=1.0 trained to s3000.
- Earlier lambda sweep included EGF lambda `0.01`, `0.1`, `1.0`; lambda `1.0` became the main candidate.

Current force-level conclusion:

- EGF lambda=1.0 improves test force metrics over EG in the P1-410 early pilot.
- Energy loss can be slightly worse than EG, so report energy-force-Hessian tradeoff instead of force-only claims.

### Full-Scale Fixed-Density Training Prep

Readiness report:

- `docs/qm9_full_scale_eg_egf_training_readiness.md`

Current full-scale assumption:

- dataset: `${DFT_DATA}/QM9PBEForceFull`;
- split: `${DFT_DATA}/QM9PBEForceFull/split.pkl`;
- labels: about `535540` geometries from `133885` molecules with reference + 3 perturbations.

DDP status:

- 8xA100 torchrun launch script exists: `scripts/launch_qm9_full_scale_8xa100.sh`;
- 4-GPU one-line full training wrapper exists: `scripts/launch_qm9_full_scale_4gpu.sh`;
- Lightning DDP config exists: `configs/ml/trainer/ddp_8xa100.yaml`;
- EG full config defaults to per-GPU batch `8`;
- EGF lambda=1.0 full config defaults to per-GPU batch `4`;
- EGF batch `8` can be tested by setting `PER_GPU_BATCH_SIZE=8`;
- `trainer.accumulate_grad_batches` is configurable through `ACCUMULATE_GRAD_BATCHES`;
- validation/test logs now use explicit `sync_dist=True`;
- full-scale EG validation uses energy loss only;
- full-scale EGF validation uses energy + force loss only;
- resume uses the existing `ckpt_path=/path/to/last.ckpt` path and the launch script passes through extra Hydra overrides.
- direct 4-GPU full-scale launches are:
  - `scripts/launch_qm9_full_scale_4gpu.sh eg 10`;
  - `scripts/launch_qm9_full_scale_4gpu.sh egf 10`;
  - `scripts/launch_qm9_full_scale_4gpu.sh egf 20`;

Remote `node01` deployment:

- base directory: `/scratch/xzh`;
- code: `/scratch/xzh/code/structures25`;
- Python environment: `/scratch/xzh/envs/structures25`;
- data root: `/scratch/xzh/data`;
- model root: `/scratch/xzh/models`;
- activation helper: `source /scratch/xzh/env.sh`;
- full 4-GPU one-line wrapper: `/scratch/xzh/run_qm9_full_4gpu.sh egf 10`;
- P1 calibration wrapper: `/scratch/xzh/run_qm9_p1_calibration_4gpu.sh egf 20`;
- setup notes on remote: `/scratch/xzh/README_qm9_setup.md`.

Remote environment check:

- Python `3.11.15`;
- PyTorch `2.4.1+cu121`;
- Lightning `2.5.0`;
- PyG CUDA extensions installed;
- PySCF `2.4.0`;
- 2-GPU NCCL all-reduce smoke passed;
- EG and EGF 4-GPU P1 `max_steps=2` smoke runs passed.

Remote data status:

- P1 pilot is staged at `/scratch/xzh/data/QM9PBEForcePilot`;
- P1 labels: `1640`;
- P1 `split.pkl` is present;
- QM9 raw xyz is staged at `/scratch/xzh/data/QM9/raw`;
- QM9 raw xyz count: `133885`;
- full dataset `/scratch/xzh/data/QM9PBEForceFull` is still missing;
- do not symlink pilot data to the full dataset name.

2026-07-08 node02 full-training attempt:

- node02 Slurm state was `IDLE`, but direct `nvidia-smi` showed GPU 1-6 occupied by two long-running `flow_grpo` reward server processes:
  - PID `2531741`, `hpsv3_scorer_multi`, `CUDA_VISIBLE_DEVICES=1,2`, about `40592 MiB` on each of GPU 1 and 2;
  - PID `2531745`, `videoalign_scorer_multi`, `CUDA_VISIBLE_DEVICES=3,4,5,6`, about `17 GiB` on each of GPU 3-6.
- node02 GPU 0 and 7 were free, but an 8-GPU job should not be started while GPU 1-6 are occupied.
- full labels were not found locally or on `/scratch`, so EG/EGF full training was not submitted.
- Remote 8-GPU launch helpers were prepared:
  - `/scratch/xzh/run_qm9_full_8gpu.sh`;
  - `/scratch/xzh/slurm_qm9_full_node02_eg.sbatch`;
  - `/scratch/xzh/slurm_qm9_full_node02_egf.sbatch`.
- Free alternative nodes observed at that time: `node04`, `node05`, `node06`.

2026-07-08 node03 request:

- Slurm node list contains `node01`, `node02`, `node04`, `node05`, and `node06`; `node03` is not registered.
- `scontrol show node node03` returns `Node node03 not found`.
- Direct SSH from the login node to `node03` fails with hostname resolution error.
- No QM9 full label generation or full EG/EGF training was started on `node03`.
- Current direct GPU checks show `node04`, `node05`, and `node06` have 8 x A100 visible with zero GPU memory used.
- Full QM9 force-label generation remains the blocker before full EG/EGF training: raw QM9 xyz exists, but `QM9PBEForceFull` labels are not present.
- Estimated full label count is `133885 * 4 = 535540`; using the P1 staged dataset size as a rough storage proxy, reserve at least `3-5 TB` before launching full generation.
- Recommended execution model is Slurm-array or multi-node sharded label generation by molecule ranges, followed by dataset validation, split/stat generation, and only then 8-GPU EG/EGF training.

2026-07-08 external Hessian QM9 Figshare dataset:

- Source: `https://figshare.com/articles/dataset/b_Hessian_QM9_Dataset_b/26363959`, DOI `10.6084/m9.figshare.26363959.v4`, license `CC0`.
- Article metadata reports `hessian_qm9_DatasetDict.zip` of `6281831499` bytes plus four `params_*.npz` files.
- Dataset contains `41645` optimized QM9 H/C/N/O molecules in each of `vacuum`, `thf`, `toluene`, and `water`.
- Expected fields are `energy`, `positions`, `atomic_numbers`, `forces`, `frequencies`, `normal_modes`, `hessian`, and `label`.
- This is an external `omegaB97x/6-31G*` Hessian/force dataset and is not a drop-in replacement for the current PBE OFDFT `QM9PBEForceFull` labels because it does not include PBE `.chk`, OFDFT density coefficients, or density-gradient labels.
- New integration script: `scripts/download_hessian_qm9_figshare.py`.
- New adoption note: `docs/qm9_figshare_hessian_dataset_adoption.md`.
- Do not symlink this dataset to `QM9PBEForceFull`; keep it under a separate root such as `/scratch/xzh/data/HessianQM9Figshare`.

2026-07-08 node04 1000 molecule demo plan:

- User requested a node04-only 8GPU demo that first computes forces/labels, then trains.
- Demo dataset name: `QM9PBEForceDemo1000`.
- Scope: 1000 QM9 molecules, reference + 3 perturbations, expected `4000` labels.
- Labelgen uses PBE `6-31G(2df,p)` with force labels, matching the existing P1 force-label route.
- Planned node04 parallelism:
  - `LABEL_NUM_PROCESSES=20`;
  - `LABEL_NUM_THREADS=1`;
  - `MAX_MEMORY_PER_PROCESS=4000 MB`;
  - 8GPU DDP training for EG and EGF lambda=1.0.
- Estimated runtime:
  - labelgen ideal: about `11.6 h`;
  - labelgen conservative: about `17-29 h`;
  - transform/statistics/training: about `1.5-6 h`;
  - Slurm wall time: `48 h`.
- New pipeline script: `scripts/launch_qm9_node04_demo1000_pipeline.sh`.
- New Slurm wrapper: `scripts/slurm_qm9_node04_demo1000_pipeline.sbatch`.
- New plan document: `docs/qm9_node04_demo1000_plan.md`.
- Local checks passed: shell syntax for both scripts and Hydra config expansion for demo datagen/training.
- Current blocker: remote SSH through the jump host times out during banner exchange, so scripts have not yet been synced to `/scratch` and no node04 job has been submitted.

2026-07-09 node04 random1000 force-label job:

- User requested starting a node04-only random 1000 molecule force/labelgen job with 20 processes, without training.
- New scripts:
  - `scripts/prepare_qm9_random_subset.py`;
  - `scripts/launch_qm9_node04_random1000_labels.sh`;
  - `scripts/slurm_qm9_node04_random1000_labels.sbatch`.
- New doc: `docs/qm9_node04_random1000_force_labelgen.md`.
- Output dataset: `/scratch/xzh/data/QM9PBEForceRandom1000`.
- Random raw symlink subset: `/scratch/xzh/data/QM9RandomSubsets/random1000_seed20260709/raw`.
- Run root: `/scratch/xzh/models/labelgen/runs/QM9PBEForceRandom1000_20260709_103746`.
- First attempt `422` failed because node04 could not resolve the venv Python symlink.
- Fixed remote environment by copying the uv CPython 3.11 runtime to `/scratch/xzh/uv-python/cpython-3.11-linux-x86_64-gnu` and repointing `/scratch/xzh/envs/structures25/bin/python*`.
- Relaunched Slurm job: `423` on `node04`.
- Final status checked 2026-07-13:
  - `COMPLETED`, exit code `0:0`;
  - start `2026-07-09T10:37:46`;
  - end `2026-07-11T10:05:05`;
  - elapsed `1-23:27:19`.
- Final artifact counts:
  - random subset raw symlinks: `1000`;
  - `.chk` files: `4000`;
  - label `.zarr.zip` files: `4000`;
  - force-label check: `4000/4000` files valid, `failures=[]`, `force_norm_max=0.13995190142033215`.
- Output size: `/scratch/xzh/data/QM9PBEForceRandom1000` is about `40G`.
- Stage completion timestamps:
  - `01_kohn_sham`: 2026-07-10 19:22;
  - `02_labelgen`: 2026-07-11 10:01;
  - `03_force_check`: 2026-07-11 10:05.
- This job generated labels only; no EG/EGF training was launched.
- Runtime used almost the full 48 h wall-time budget, so future random-1000 force-label jobs should request `60-72 h` or increase parallelism after checking memory.

2026-07-13 random1000 8xA100 EG/EGF training:

- New doc: `docs/qm9_random1000_8xa100_training.md`.
- New scripts:
  - `scripts/launch_qm9_random1000_train_8xa100.sh`;
  - `scripts/slurm_qm9_random1000_train_8xa100.sbatch`.
- `node04` was occupied by an existing non-Slurm 8-GPU job, so the training was run on free 8xA100 `node05`.
- Dataset: `/scratch/xzh/data/QM9PBEForceRandom1000`.
- Label files: `4000`; cached transformed labels: `4000`; `.chk` files under `kohn_sham`: `4000`.
- Split:
  - label-file split: train `3200`, val `400`, test `400`;
  - expanded sample sizes in `split["sizes"]`: train `42670`, val `5404`, test `5373`.
- Slurm job `479` completed:
  - state `COMPLETED`, exit code `0:0`;
  - node `node05`;
  - start `2026-07-13T16:28:28`;
  - end `2026-07-13T17:17:34`;
  - elapsed `00:49:06`.
- Run root: `/scratch/xzh/models/train_random1000/20260713_162829`.
- EG run:
  - directory: `/scratch/xzh/models/train/runs/qm9_random1000_eg_e10_20260713_162829`;
  - best checkpoint: `checkpoints/epoch_009.ckpt`;
  - train time: `15:44.80`;
  - mean throughput: `521.53` samples/s;
  - final validation: energy loss `0.02444537`, total `0.00249360`.
- EGF lambda=1.0 run:
  - directory: `/scratch/xzh/models/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829`;
  - best checkpoint: `checkpoints/epoch_009.ckpt`;
  - train time: `33:19.14`;
  - mean throughput: `218.87` samples/s;
  - final validation: energy loss `0.02273401`, force loss `0.00388426`, total `0.00268172`.
- Force path remains scalar energy autograd: `F_pred = -dE_pred/dR`.
- No density-relaxed Hessian evaluation was run in this training job.
- No force head was added.
- Completed job throughput JSON used the old `estimate_num_samples=4000` field; the launcher has since been patched to use `split["sizes"]["train"]` for future estimate reporting.

2026-07-13 random1000 EG/EGF energy/force/Hessian evaluation:

- New doc: `docs/qm9_random1000_eg_egf_energy_force_hessian_eval.md`.
- New scripts:
  - `scripts/launch_qm9_random1000_eval.sh`;
  - `scripts/slurm_qm9_random1000_eval.sbatch`.
- Updated eval scripts:
  - `scripts/qm9_force_eval.py` now reports energy MAE/RMSE alongside force metrics;
  - `scripts/qm9_pbe_hessian_reference_set.py` can resolve random1000 `.chk` names;
  - `scripts/qm9_hessian_eval_reference_set.py` loads checkpoints on CPU before moving to GPU.
- Slurm job `480` completed:
  - state `COMPLETED`, exit code `0:0`;
  - node `node05`;
  - elapsed `00:59:41`.
- Output directory: `/scratch/xzh/models/eval/qm9_random1000_eg_egf_metrics/20260713_190645`.
- Energy/force test coverage: `4973` samples, `87075` atoms, `0` failures.
- Energy is the configured `e_kin_plus_xc` label.
- Energy/force summary:
  - EG energy MAE/RMSE: `0.027565` / `0.038543`;
  - EGF lambda=1.0 energy MAE/RMSE: `0.027027` / `0.037877`;
  - EG force component MAE/RMSE: `0.070219` / `0.094251`;
  - EGF lambda=1.0 force component MAE/RMSE: `0.003678` / `0.006132`.
- Hessian summary:
  - generated `10/10` PBE analytic Hessian references from random1000 test molecules;
  - model Hessian is fixed-density finite difference of `F_pred = -dE_pred/dR`, displacement `1e-3`, `scf_iteration=1`;
  - EG Hessian MAE/RMSE/relative Fro: `0.075317` / `0.240548` / `2.256694`;
  - EGF lambda=1.0 Hessian MAE/RMSE/relative Fro: `0.019962` / `0.059027` / `0.552192`.
- EGF improves force and fixed-density Hessian proxy strongly; energy MAE/RMSE is also slightly better, but energy max absolute error is higher.
- This is not a density-relaxed physical Hessian result.

2026-07-14 random1000 Test100 GPU4PySCF Hessian extension:

- `scripts/qm9_pbe_hessian_reference_set.py` now supports `--backend cpu|gpu4pyscf`.
- GPU4PySCF mode is externally sharded with one process per CUDA device; the script rejects
  `--workers` values other than `1` for this backend.
- Five-molecule GPU4PySCF-vs-CPU PySCF validation passed:
  - output: `/scratch/xzh/models/eval/qm9_random1000_gpu4pyscf_hessian_validation/20260714_131826`;
  - maximum relative Frobenius error: `2.179982e-4`;
  - maximum absolute element error: `3.416659e-4`;
  - configured thresholds: `1e-3` and `2e-3`, respectively;
  - GPU elapsed time: `29.98-53.71 s` per molecule, mean about `39.8 s`.
- CPU full-test job `483` was cancelled after the GPU backend passed validation. It had produced
  `62` reusable PBE Hessian cache files; no cache was deleted.
- Job `484` was an incomplete `80/100` debug run. Cause: each of eight GPU shards inherited
  `qm9_pbe_hessian_reference_set.py`'s default `--max-molecules=10`.
- The launcher now computes each shard size and passes it explicitly through `--max-molecules`.
- Final job `485` completed on node04 with 8 x A100:
  - output: `/scratch/xzh/models/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/20260714_135600`;
  - summary: `summary.json` under that output directory;
  - shared PBE cache: `/scratch/xzh/models/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians`;
  - PBE reference coverage: `100/100`, `0` failures, `92` cached at stage start, `8` newly computed;
  - model Hessian coverage: EG `100/100`, EGF lambda=1.0 `100/100`, no failures;
  - PBE stage wall time: `168 s`; model fixed-density stage wall time: `305 s`;
  - EG Hessian MAE/RMSE/relative Fro/symmetry: `0.066706` / `0.200525` / `2.921985` / `0.000408`;
  - EGF Hessian MAE/RMSE/relative Fro/symmetry: `0.010798` / `0.034852` / `0.479906` / `0.000129`;
  - EGF improves MAE by about `6.18x`, RMSE by `5.75x`, and relative Frobenius by `6.09x`.
- The EG/EGF ordering is stable from the earlier 10-molecule subset to all 100 test molecules.
- This remains a fixed-density finite-difference Hessian proxy at `scf_iteration=1`, not a
  density-relaxed physical Hessian.
- No labels, checkpoints, PBE Hessian references, or cached model artifacts were deleted.

2026-07-14 random1000 Test100 density-relaxed derived-force Hessian:

- New report: `docs/qm9_random1000_test100_density_relaxed_hessian.md`.
- Slurm job `486` completed on node04 with 8 x A100:
  - state `COMPLETED`, exit code `0:0`;
  - start `2026-07-14T15:05:45`, end `2026-07-14T19:17:45`;
  - Slurm elapsed `04:12:00`; measured evaluator wall time `15116 s` (`4:11:56`);
  - output: `/scratch/xzh/models/eval/qm9_random1000_test100_density_relaxed_hessian_node04/20260714_150546`.
- Configuration matches the validated strict P1 route:
  - base-density warm-start from `sad_default`;
  - stage 1 Adam `lr=1e-3`, `max_cycle=1000`, threshold `1e-2`;
  - stage 2 Adam `lr=3e-4`, `max_cycle=10000`, threshold `1e-4`;
  - `fallback-always`; finite-difference displacement `1e-3` Bohr.
- Convergence:
  - base density: EG `100/100`, EGF `100/100` strict;
  - displaced points: EG `10638/10638`, EGF `10638/10638` strict;
  - no failed molecules, non-finite forces, or missing Hessians;
  - 200 density-relaxed Hessian NPZ files were saved.
- Density-relaxed metrics against PBE:
  - EG MAE/RMSE/relative Fro/symmetry: `0.070755` / `0.207097` / `3.046408` / `0.444027`;
  - EGF lambda=1.0: `0.012074` / `0.037043` / `0.516296` / `0.126402`;
  - EGF improves MAE by `5.86x`, RMSE by `5.59x`, and relative Frobenius by `5.90x`;
  - EGF is better on MAE/RMSE/relative Fro for every `100/100` molecule.
- Density relaxation makes agreement with PBE modestly worse than fixed density for both models:
  - EG MAE/RMSE/relative Fro increase by `6.1%` / `3.3%` / `4.3%`;
  - EGF increases by `11.8%` / `6.3%` / `7.6%`.
- Symmetry error increases strongly after density relaxation, indicating that residual optimization
  noise is amplified by the `1e-3` finite difference even at strict projected-gradient threshold.
- Time:
  - EG mean molecule time `465.84 s`; EGF `671.37 s`, a `1.44x` ratio;
  - allocated 8-GPU time about `33.59 GPU-hours`;
  - strict density-relaxed wall time is about `49.6x` the earlier fixed-density FD stage (`305 s`);
  - runtime is primarily CPU PySCF setup/density optimization bound, not A100 memory bound.
- Hard cases:
  - largest EG error: `0040728`, MAE `0.245101`, relative Fro `13.5344`;
  - largest EGF error: `0000777`, MAE `0.032716`;
  - slowest EG: `0015263`, `813.5 s`; slowest EGF: `0060531`, `1062.8 s`.
- This supports a strict-converged density-relaxed derived-force EGF ordering improvement, but it
  is not an analytic total-OFDFT physical Hessian result because classical nuclear derivatives are
  absent from the current force path.

Autograd status:

- EG does not compute force autograd because `force_supervision=false`;
- EGF force remains `F_pred = -dE_pred/dR`;
- force `create_graph=True` is used only during training when active nonzero `force_loss` exists;
- validation/test force uses `create_graph=False`;
- full-scale validation disables default density-gradient metrics with `metric_interval=0`;
- full-scale validation uses a separate `validation_loss_function`;
- density-gradient training still requires higher-order autograd through `dE/dcoeffs`.

Throughput monitoring:

- callback: `mldft.ml.callbacks.ThroughputMonitor`;
- CSV output: `${paths.output_dir}/throughput/throughput.csv`;
- JSON output: `${paths.output_dir}/throughput/throughput_summary.json`;
- records samples/sec, steps/sec, data loading time, forward time, force autograd time, backward time, optimizer time, peak GPU memory, and 8-GPU time estimates.

Smoke results:

| check | result |
| --- | --- |
| full EG/EGF config compose | pass |
| `py_compile` for model and throughput callback | pass |
| launch script `bash -n` | pass |
| `pytest tests/ml/test_mldft_module.py -q` | 7 passed |
| `pytest tests/ml/test_train.py::test_train_resume -q` with env vars | 1 passed |
| EG 1-GPU calibration smoke | pass, 4 steps |
| EGF 1-GPU calibration smoke | pass, 4 steps |
| EG 2-GPU DDP smoke | pass, 2 steps |
| EGF 2-GPU DDP smoke | pass, 2 steps |
| EG validation smoke | pass, 1 train + 1 val |
| EGF validation smoke | pass, 1 train + 1 val |
| EGF validation-only smoke | pass, energy + force only |

Known full-scale prep limits:

- `QM9PBEForceFull` was not present, so full-scale data I/O has not been tested.
- Tiny smoke throughput is pessimistic and should not be used for scheduling a full job.
- A realistic 8-GPU calibration should run 50-200 measured steps before launching 10/20 epochs.
- DDP emitted a grad-stride performance warning; monitor this if scaling is poor.

### Fixed-Density / Proxy Hessian

Earlier finite-difference fixed-density Hessian probes showed EGF lambda=1.0 better than EG on small held-out subsets.

Current status:

- finite-difference fixed-density Hessian is still the official stable evaluation path;
- second-order autograd fixed-density Hessian now works on the tested 10 molecule reference subset after the self-loop fix;
- 20 molecule fixed-density autograd validation is still pending.

### Density Optimization

Initial blocker:

- `sad_default + SGD lr=1e-3 + max_cycle=50 + threshold=1e-4` did not converge on the 5 original molecules.

Resolved setting:

- initialization: `sad_default`
- optimizer: Adam
- first stage: `lr=1e-3`, `max_cycle=1000`, threshold `1e-2`
- second stage/fallback: `lr=3e-4`, `max_cycle=10000`, threshold `1e-4`
- base-density warm-start enabled
- fallback always enabled

This reduced the 5-molecule density-relaxed Hessian wall time from about `3:08:19` to about `17:01.99` while preserving the Hessian conclusion.

### Density-Relaxed Hessian

10-molecule confirmatory run:

- output prefix: `_runtime/qm9_p1_models/eval/qm9_p1_410_density_relaxed_hessian_10mol/density_relaxed_10mol_basewarm_twostage_strict`
- 924/924 displaced density optimizations reached strict convergence `<1e-4`;
- wall time: `33:30.03`;
- max RSS: `1.63 GB`.

Mean 10-molecule Hessian metrics:

| model | Hessian MAE | Hessian RMSE | relative Fro | symmetry max abs |
| --- | ---: | ---: | ---: | ---: |
| EG_s3000 | 1.192573 | 3.588587 | 21.397097 | 13.720957 |
| EGF_lam1_s3000 | 0.041654 | 0.106804 | 0.834243 | 0.414696 |

Current density-relaxed conclusion:

- On this 10-molecule P1-410 early-pilot density-relaxed derived-force Hessian benchmark, EGF lambda=1.0 is clearly better than EG.
- This is still not a final P1/P2 conclusion.

### Second-Order Autograd Hessian

Feasibility audit found the initial blocker:

- full-edge second-order autograd Hessian/HVP produced NaN;
- removing self-loop edges made autograd finite;
- root cause: `torch.norm(0)` on self-loop edge has NaN second derivative.

Fix completed:

- self-loop distance now uses constant zero through `_safe_edge_lengths()`.

Post-fix audit:

- output: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_self_loop_fix/`
- wall time: `34.44 s`;
- max RSS: `1.75 GB`;
- exit status: 0.

Autograd-vs-FD fixed-density Hessian after fix:

| model | molecule | relative Fro |
| --- | ---: | ---: |
| EG_s3000 | 0000010 | 6.80e-4 |
| EG_s3000 | 0000062 | 5.19e-3 |
| EGF_lam1_s3000 | 0000010 | 8.06e-5 |
| EGF_lam1_s3000 | 0000062 | 9.49e-5 |

HVP status:

- HVP is finite after the self-loop fix;
- HVP agrees with finite-difference directional response on `0000010` and `0000062`;
- HVP is a good next primitive for fixed-density experiments and possible future implicit Hessian, but not yet the official evaluation path.

10 molecule validation:

- output: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_10mol_validation/`
- molecules: `0000010`, `0000323`, `0000109`, `0000171`, `0000062`, `0000116`, `0000144`, `0000019`, `0000052`, `0000350`;
- wall time: `1:01.60`;
- max RSS: `1.75 GB`;
- full-edge fixed-density Hessian: 20/20 rows finite;
- HVP: 40/40 rows finite;
- EG/EGF ordering vs PBE reference is consistent between autograd and FD on 10/10 molecules.

10 molecule autograd-vs-FD fixed-density Hessian aggregate:

| model | MAE mean | RMSE mean | relative Fro mean | relative Fro max |
| --- | ---: | ---: | ---: | ---: |
| EG_s3000 | 2.801e-4 | 1.666e-3 | 1.137e-3 | 5.190e-3 |
| EGF_lam1_s3000 | 2.875e-6 | 8.730e-6 | 7.428e-5 | 1.187e-4 |

### Fixed-Density vs Density-Relaxed Hessian and Time Cost

Comparison report:

- `docs/qm9_p1_410_fixed_vs_density_relaxed_hessian_time_comparison.md`

Artifacts:

- fixed-density autograd/FD source: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_10mol_validation/`
- density-relaxed matrix rerun: `_runtime/qm9_p1_models/eval/qm9_p1_410_fixed_vs_density_relaxed_hessian_time_comparison/`
- PBE 10 molecule manifest: `_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessian_reference_manifest.json`

Code note:

- `scripts/qm9_hessian_density_relaxed_eval.py` now has optional `--hessian-npz-dir` to save density-relaxed Hessian matrices for direct fixed-vs-relaxed comparison.

Main 10 molecule conclusion:

- Fixed-density autograd is a fast screening proxy, not a matrix-equivalent replacement for density-relaxed Hessian.
- Fixed-density and density-relaxed rankings agree: EGF lambda=1.0 is better than EG on 10/10 molecules in both paths.
- Density relaxation does not move these checkpoints closer to PBE on this set. Versus fixed-density autograd, density-relaxed MAE is worse on 9/10 EG molecules and 10/10 EGF molecules.
- Important hard cases: `0000323` for EG density-relaxed Hessian, `0000052` for EGF fixed-vs-relaxed deviation.

Mean 10 molecule comparison:

| model | fixed auto vs PBE MAE | density-relaxed vs PBE MAE | fixed auto vs relaxed MAE | fixed auto vs PBE rel Fro | density-relaxed vs PBE rel Fro |
| --- | ---: | ---: | ---: | ---: | ---: |
| EG_s3000 | 0.418686 | 1.192575 | 0.836616 | 12.802920 | 21.397202 |
| EGF_lam1_s3000 | 0.019494 | 0.041653 | 0.026179 | 0.392696 | 0.834249 |

Timing:

| scope | wall / sum time | max RSS | note |
| --- | ---: | ---: | --- |
| PBE reference set | `53:54.87` wall; `6324.8 s` manifest sum | 2.18 GB | workers=2; `0000010` cached |
| fixed-density autograd validation | `1:01.60` wall; `7.35 s` autograd row sum | 1.75 GB | both models; script also ran FD/HVP |
| density-relaxed matrix rerun | `31:13.83` wall; `1845.2 s` row sum | 1.74 GB | both models, strict `<1e-4` |

Operational reading:

- fixed-density autograd is about 30x faster than density-relaxed by wall time for the validation scripts;
- density-relaxed OFDFT shows an operational speed advantage over the PBE reference in this small run, but the comparison is not fully controlled.

## 2026-07-15 Conservative total-OFDFT and model-candidate update

The previous random1000 density-relaxed Hessians differentiate only the learned `kin_plus_xc`
force and remain `incomplete-derived-force Hessian proxies`. A complete constrained scalar total
energy, overlap/Pulay derivative, strict stationary-density solver, scalar-derived total force,
KKT response HVP and strict full-Hessian evaluator are now implemented. Main code:

- `mldft/ofdft/{conservative_force,geometry_integrals,implicit_response,stationary_density}.py`;
- tensor updates in `basis_integrals.py`, `energies.py`, and `functional_factory.py`;
- `scripts/qm9_total_ofdft_{force,hvp,hessian}_audit.py`;
- `scripts/qm9_hessian_vibrational_metrics.py`;
- `scripts/qm9_total_ofdft_model_optimization_analysis.py`.

The complete force passes relaxed scalar first derivatives, a `-7.76e-10 Ha` closed loop at
`1e-5 Bohr`, and cross-Hessian/KKT checks at roughly `1e-4` relative on P1 `0000010`. Full local
Hessians are stable between `1e-5` and `3e-5 Bohr`. Very small scalar-energy second differences
are cancellation dominated; use `3e-3--1e-3` for the energy-curvature audit while independently
checking a local total-force-difference interval.

Model ablation status:

- scalar-energy-secant-only: rejected; force component MAE is about 39x worse;
- actual force weight 3: rejected as default complete-total candidate because `0003027` regresses
  strongly despite two other directional improvements;
- actual force weight 1 plus scalar secant weight 0.01: retained candidate.

Retained candidate:

```text
/scratch/xzh/models/train/runs/
  qm9_random1000_forcew1p0_secantw0p01_e10_20260715_210000/
  checkpoints/epoch_009.ckpt
```

Slurm 880 completed 10 epochs on eight A100 GPUs in `45:58`. Test100 energy/force component MAE
changes `0.027571/0.002463 -> 0.021752/0.002177`; it wins 10/10 frozen fixed-density proxy
Hessians. On five strict complete-total directional checks it wins 4/5, changes mean MAE/RMSE
`0.265930/0.665832 -> 0.235458/0.574531`, and regresses 9.46% on `0003027`. Full `0000777`
MAE/RMSE/relative Frobenius changes `0.127711/0.287738/2.05393 ->
0.099883/0.211838/1.51214`; all 42/42 points are strict and the result is stable at `3e-5`.

Frozen coordinate-0 Test100 complete-total HVP is also complete: candidate wins 92/100 and lowers
mean MAE/RMSE/relative Frobenius from `0.191748/0.572946/10.9965` to
`0.139365/0.423238/9.5081`. Slurm 933 used 16 workers/eight A100 GPUs, finished in `56:44`, and
had 38.6 GiB MaxRSS. This accepts the checkpoint as the leading random1000 candidate, while the
eight regressions listed in the primary report remain mandatory controls.

Primary report and aggregate artifact:

```text
docs/qm9_random1000_total_ofdft_hessian_model_optimization.md
/scratch/xzh/models/eval/qm9_random1000_force_secant_total_analysis/20260715
```

The artifact contains per-molecule CSVs, training/curvature/HVP plots, checkpoint provenance and
both full-Hessian step checks. Do not claim a uniform complete-total improvement: `0003027` is a
documented regression and the full matrix/frequency comparison currently has one anchor molecule.

## HVP100 Direct Curvature Supervision (2026-07-16)

This experiment asks whether sparse direct HVP supervision can improve validation force and
strict density-relaxed complete-total curvature simultaneously. Test100 is frozen: only its parent
IDs were used for a zero-overlap assertion. The frozen source split SHA256 is
`919c5f6c250b0d2e9f14f43522894fcac8b179d6d4fd5b774c2cec1ea249034b`; train100 selection SHA256
is `8982fc6349b520b9b98f52fa684e81558b6afab16d14a100f3c3843608503f0b`. The resulting split has
100/100/100 train/validation/test parents and zero parent overlap. All six train geometries follow
each selected parent.

Training references and definition:

- GPU4PySCF PBE analytic Hessians succeeded for all 100 train and 20 representative validation
  parents; combined manifest SHA256
  `bee31003d563e8cd2fe516a6e90b2255d228b8ad6446a6b026dfd8005bfdc672`;
- 120 deterministic directions comprise 42 paired, 44 random internal, 23 bond-stretch and 11
  low-mode directions after translation/rotation removal; sidecar manifest SHA256
  `4ac88f11740a9f35e9cdc0cbdbb9e27cb6064db08864dff3d77f36249469b1cd`;
- the training target is explicitly a learned-energy fixed-density HVP compared to PBE analytic
  total `H.v`, not a relaxed complete-total OFDFT HVP;
- strict validation uses finite differences of complete scalar-derived total force after strict
  density relaxation at both directional endpoints.

New training code/configuration includes HVP sidecar loading in `OFData`,
`DirectionalHVPLoss`, scalar-energy HVP autograd in `MLDFTModule`, HVP throughput timing,
`configs/ml/experiment/str25/qm9_pbe_force_full_egf_hvp.yaml`, the secant+HVP counterpart and
`scripts/launch_qm9_hvp100_train.sh`. There is no force head. Direct HVP is eligible on 25% of
batches and actually ran on 11.12% of formal batches/3.15% of graphs. Seven focused loss/autograd
tests and an eight-step real-GPU checkpoint smoke pass.

Formal job 1021 completed 24 one-A100 runs: A energy+force, B plus secant, C plus direct HVP and D
plus both; C/D use weights `1e-5/1e-4/1e-3`, all with three matched seeds and 600 optimizer steps.
No stage-1 model passes the joint gate. C `1e-3` has the best fixed-HVP change (`-1.12%`) but force
and energy regress `5.12%` and `13.17%`. Direct HVP adds 6--10% training wall time and 46% peak
allocated GPU memory at the measured activation rate; all 24 runs consumed 6.65 A100 GPU-hours.

Strict validation20 completed all 240 model/molecule tasks. C `1e-3` improves complete-total HVP
MAE by `14.51%`, all 3/3 seeds improve, and the molecule win fraction is `0.55`. It still fails:
only 1/3 seeds improves force and 1/3 passes the 5% energy gate. D `1e-5` worsens strict HVP by
`8.08%`. Hard case `0044504` dominates the aggregate, while the small-reference case `0059830`
also regresses under direct HVP. Gradient-conflict measurements do not show systematic HVP-force
opposition; sparse directions and the surrogate label definition are the leading concerns.

The diagnostic-only matched-seed full-Hessian audit completed 9/9 matrices and all `846/846`
displaced points strict. On small `0000751`, median `0103559` and hard `0044504`, C `1e-3`
changes mean complete-total Hessian MAE/RMSE/relative Frobenius from
`26.1366/163.729/2590.11` to `23.1422/143.721/2273.55`, and wins all three molecule MAEs.
This is not promotable evidence: `0044504` has asymmetry ratio about `0.29` and dominates the
average even at `1e-8` density residual. Frequency MAE changes only `6071.5 -> 5957.9 cm^-1`, while
the model still has 106 imaginary modes versus seven for PBE. Excluding the hard case, frequency
RMSE is essentially unchanged. The nine matrices cost 3.88 A100 GPU-hours with a 51:50 array
makespan. Raw point, matrix, resource and vibrational tables are under
`20260716_gated25v2/full_hessian/`.

Formal root and local small-file mirror:

```text
/scratch/xzh/models/hvp100/20260716_gated25v2
_runtime/remote_artifacts/qm9_hvp100
```

No candidate is frozen, so do not run a new Test100 confirmation and do not expand to 400--800
parents. The next model iteration should first align training HVP labels with the response-aware
complete-total definition, increase validation directional coverage, and retune normalization on
validation only.

## HVP Curvature Improvement v1 In Progress (2026-07-16)

The active experiment replaces the failed one-direction HVP100 fine-tune with train800
energy/density-gradient/force replay plus strict-stable train100 curvature supervision. Test100 is
forbidden until a candidate and promotion rule are frozen. Machine-readable preregistration:

```text
configs/audit/qm9_hvp_branch_stability_v2.yaml
configs/audit/qm9_hvp_branch_stability_v3.yaml
configs/audit/qm9_hvp_curvature_training_v2.yaml
configs/audit/qm9_hvp_curvature_validation_v1.yaml
```

The branch audit uses four directions per parent (random internal, bond stretch, angle bend and
low frequency), three density branches, strict density/KKT/response gates, HVP steps
`3e-5/1e-5/3e-6 Bohr`, and a separate `1e-3 Bohr` scalar-energy closure check. V2 keeps the
strict all-four parent result as mandatory sensitivity evidence. The preregistered train12 pilot
found 19/48 stable directions but 0/12 all-four parents, with several clean parents having only a
direction-specific low-frequency FD failure. V3 therefore changes aggregation, not numerical
gates: any density/branch/KKT failure excludes the whole parent; response/step/closure failures
mask only that direction; at least 3/4 directions must pass. The same train12 pilot gives 3/12 v3
eligible parents. `0044504` remains parent-excluded because it fails a density gate.

Persistent remote artifacts and paused Slurm history:

```text
/scratch/xzh/models/hvp_branch_stability/20260716/multidirection_sidecars
/scratch/xzh/models/hvp_branch_stability/20260716/formal_v2             # val8, job 1347
/scratch/xzh/models/hvp_branch_stability/20260716/formal_train100_v2    # jobs 1443, 1464
```

The train100 task manifest has 1200 tasks and SHA256
`bc658ab5594f30f4dc9a92080d1f64f9ef61f80477de9afa9a6b34897113dacd`.
Formal val8 is now complete with 19/32 stable directions and no failed tasks. V2 admits 2/8 parents
(`0027926`, `0132890`), while v3 admits 3/8 (`0000751`, `0027926`, `0132890`).
`0044504` remains parent-excluded by its density gate. Train100 jobs 1443/1464 were cancelled at
830/1200 validated complete tasks; 370 are frozen as missing and 40 of those have partial
directories. Jobs 1629, 1797 and 1865 were cancelled before execution. The queue is empty and A--E
training never started. See `docs/qm9_hvp_curvature_pause_node01_20260716.md`.

A later read-only capacity check over 59 fully complete train parents found 20 v3-stable parents
and 115/236 stable directions with zero load errors. It confirms enough likely curvature capacity
for the preregistered five-parent minimum, but is not the final train100 manifest or a selection
result.

The formal train100 arrays were changed from fixed 20/20 throttles to a maximum of 40 each after
their queues became imbalanced. Cluster capacity still limits aggregate concurrency to 40, while
allowing dynamic 23/17-style rebalancing. No task or numerical setting changed.

For `0044504`, branch energy/density/force agreement and response residual are excellent; the
failure is direction-specific second-response/FD instability. Random/low-frequency step metrics
are `8.53/134.7`, implicit-vs-relaxed metrics `3.86/299`, and energy-force closure mismatches
`0.995/0.999`. Its low-frequency density gradient `1.06e-8` just exceeds the frozen `1e-8`
parent gate. Do not describe this sample simply as an electronic branch crossing.

Training changes:

- `ParentPairBatchSampler` supports strict-stable pair/HVP parent filters, deterministic
  multi-direction epoch resampling and an interleaved four-batch replay cycle: 25% pair, 25% HVP,
  50% ordinary-only. This keeps the expected ordinary train800 graph fraction at 81.25%.
- direct HVP supports warm-up, ramp, alternating batches, soft per-graph caps, sparse per-loss
  gradient norms/cosines and GradNorm-style HVP/force balancing;
- before any A--E screen metric existed, train700 forgetting was made a hard gate: energy and
  force MAE must each stay within +5% of matched-seed A in all three seeds. Fixed-HVP hyperparameter
  ranking was restricted to v3-stable validation parents/directions and now depends on val20
  postprocessing. At least five stable validation parents are required for fixed-HVP ranking and
  strict HVP. Final pre-screen hashes are training v2 `f6244126...f30d91` and validation v1
  `df78b5fb...352251`; the 20-run training TSV remained byte-identical;
- variant D uses `PairRelaxedForceSecantLoss`: scalar-energy endpoint forces are matched through a
  normalized secant to complete-total PBE forces after KS density relaxation;
- variant E is explicitly a surrogate, not end-to-end implicit differentiation. Up to ten
  lexicographically first v3-eligible parents receive three-branch-mean baseline OFDFT implicit
  complete-total HVP teacher targets, while prediction remains a scalar learned-energy HVP;
- end-to-end implicit training remains blocked by detached geometry-integral preparation, density
  optimization and SciPy MINRES. Do not relabel the E surrogate as an implicit prediction path.
- strict complete-total HVP tasks are isolated by run, direction, and molecule. Strict and full
  Hessian postprocessors run after any array outcome, retain successful raw records, emit explicit
  failure CSVs, and block promotion for every incomplete run. This prevents one pathological
  response from discarding audit evidence or producing a partial-data candidate;
- multiseed reuse requires `last.ckpt` plus throughput `global_step == max_steps`; a partial
  checkpoint is not accepted as a completed seed. The node01 resume orchestrator passes the new
  val20 postprocessing job ID into the strict dependency chain and cannot reuse cancelled 1797.
- branch-stability postprocessing stores NPZ paths rather than 1200 lazy `NpzFile` handles and
  materializes only one direction's three branches at a time. A 682-record formal read-only check
  kept open file descriptors at 4 throughout, preventing an end-of-audit `EMFILE` failure.
- the downstream implicit-teacher builder now reads `implicit_hvp` by path under a context
  manager. A real val8 smoke produced a finite `(4, 9, 3)` target for `0000751` with 3/4 stable
  directions; formal train100 output directories were still clean/absent before postprocessing.
- every filtered stable-sidecar directory now has an authoritative manifest containing parent ID,
  stability mask and file SHA256. A real val8 run produced 3/3 matching entries; train100
  postprocessing blocks submission when manifest and NPZ counts differ.
- implicit teacher manifests also contain per-file hashes/masks and hashes of the source stability
  summary and parent/direction CSVs. Stable and implicit manifest counts are both checked before
  the A--E screen can be submitted.

Main new files:

```text
scripts/qm9_hvp_branch_stability_analysis.py
scripts/qm9_build_implicit_target_sidecars.py
scripts/launch_qm9_hvp_stable_replay_train.sh
scripts/qm9_hvp_curvature_stage1_analysis.py
scripts/prepare_qm9_hvp_curvature_multiseed.py
scripts/prepare_qm9_hvp_curvature_strict_tasks.py
scripts/qm9_hvp_curvature_strict_analysis.py
scripts/qm9_hvp_curvature_full_hessian_analysis.py
configs/ml/model/loss_function/l1_force_relaxed_force_secant.yaml
configs/ml/experiment/str25/qm9_pbe_force_full_egf_relaxed_force_secant.yaml
configs/ml/experiment/str25/qm9_pbe_force_full_egf_implicit_target_hvp.yaml
```

The postprocessing chain now submits the complete single-seed A--E screen, matched validation and
train700 forgetting checks, one frozen setting per C/D/E across all three seeds, and stable-only
strict complete-total HVP after val20 stability finishes. The strict protocol uses label/reference
base density, base continuation, `h=1e-5 Bohr`, staged Adam and LBFGS/Newton refinement; numerical
density/response convergence is a hard gate. No Test100 path is part of this chain.

Focused remote tests currently pass: replay sampler 7/7, HVP/relaxed-force-secant loss 4/4,
alternate HVP target loader 6/6, and active/inactive HVP backward 5/5 including existing force
tests. New screen/multiseed and strict pipeline tests pass 4/4. A--E training and validation metrics
do not exist yet; do not infer a candidate from this infrastructure milestone.

Real one-A100 smoke validation then found two launch issues before formal training: Hydra rejected
new optional dataset keys without `++`, and max-step runs could leave `last.ckpt` at an earlier
epoch. `scripts/launch_qm9_hvp_stable_replay_train.sh` now uses add-or-override dataset keys and
step checkpointing every 200 steps (step 1 in smoke). Jobs 1925/1926 validated D/E dataloading,
backward and validation; jobs 1942/1943 forced C/E HVP active from step zero and observed nonzero
HVP losses; job 1969 wrote `last.ckpt`, throughput, four gradient-norm series and six gradient
cosines. The smoke force-vs-HVP cosine was `-0.143`; it confirms conflict logging, not a converged
scientific result. `scripts/qm9_hvp_curvature_training_diagnostics.py` exports these diagnostics for
every formal screen and three-seed run.

The first 38 fully audited train100 parents give an interim 66/152 stable directions, 1/38 v2
parents and 9/38 v3 parents. Energy-force closure and step stability dominate direction masks;
density-gradient failure dominates parent exclusions. This interim snapshot is capacity evidence
only and is not used for training-sidecar freeze or model selection.

## Tests and Verification Commands

Fast local tests:

```bash
.venv/bin/python -m pytest tests/ml/test_gbf_module.py -q
```

Expected current result:

```text
9 passed
```

Runtime checkpoint test, skipped by default:

```bash
.venv/bin/python -m pytest tests/ml/test_qm9_second_order_autograd_runtime.py -q
```

Expected default result:

```text
1 skipped
```

Opt-in runtime test:

```bash
MLDFT_RUN_RUNTIME_TESTS=1 MLDFT_RUNTIME_DEVICE=cuda:0 CUDA_VISIBLE_DEVICES=4 \
  DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
  DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
  .venv/bin/python -m pytest tests/ml/test_qm9_second_order_autograd_runtime.py -q
```

Expected current result:

```text
1 passed
```

Second-order fixed-density audit:

```bash
OUT=_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_self_loop_fix
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=4 \
  .venv/bin/python scripts/qm9_second_order_autograd_hessian_audit.py \
  --molecules 0000010,0000062 \
  --scf-iteration 1 \
  --run EG_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt \
  --run EGF_lam1_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt \
  --output-dir "$OUT" \
  --output-json "$OUT/second_order_self_loop_fix_audit.json" \
  --device cuda:0 \
  --no-run-unrolled
```

## Detailed Reports

Read these for full context:

- `docs/qm9_force_hessian_p1_p2_taskbook.md`
- `docs/qm9_p1_410_training_smoke.md`
- `docs/qm9_p1_410_early_pilot_report.md`
- `docs/qm9_p1_410_hessian_reference_mini_benchmark.md`
- `docs/qm9_p1_410_lambda1_extended_hessian_validation.md`
- `docs/qm9_p1_410_hessian_physical_eval_audit.md`
- `docs/qm9_p1_410_density_optimization_convergence_sweep.md`
- `docs/qm9_p1_410_converged_density_relaxed_hessian_eval.md`
- `docs/qm9_p1_410_bad_displacement_rescue.md`
- `docs/qm9_p1_410_density_relaxed_hessian_cost_reduction.md`
- `docs/qm9_p1_410_density_relaxed_hessian_10mol.md`
- `docs/qm9_p1_410_second_order_autograd_hessian_feasibility.md`
- `docs/qm9_p1_410_second_order_autograd_self_loop_fix.md`
- `docs/qm9_p1_410_second_order_autograd_10mol_validation.md`
- `docs/qm9_p1_410_fixed_vs_density_relaxed_hessian_time_comparison.md`
- `docs/qm9_full_scale_eg_egf_training_readiness.md`
- `docs/gpu4pyscf_force_label_migration.md`
- `docs/qm9_node04_random1000_force_labelgen.md`
- `docs/qm9_random1000_8xa100_training.md`
- `docs/qm9_random1000_eg_egf_energy_force_hessian_eval.md`
- `docs/qm9_random1000_test100_density_relaxed_hessian.md`
- `docs/qm9_random1000_density_relaxed_hessian_diagnosis_and_optimization.md`
- `docs/qm9_total_ofdft_conservative_force_hessian.md`
- `docs/qm9_random1000_total_ofdft_hessian_model_optimization.md`
- `docs/qm9_random1000_hvp100_curvature_supervision.md`
- `docs/qm9_random1000_hvp_curvature_improvement_v1.md`
- `docs/qm9_structured_density_hessian_head_pilot_v1.md`
- `docs/qm9_graphformer_frozen_hessian_attention_head_pilot_v1.md`
- `docs/qm9_graphformer_frozen_force_attention_pilot_v1.md`

## Complete-Total Capacity Update

The active capacity-ceiling experiment is documented in
`docs/qm9_complete_total_hessian_capacity_v1.md`. It remains train-only and has consumed zero
validation parents and zero Test100 records.

- Original five strict baseline relative Frobenius spans `1.29--6.60`; density gradients are about
  `1e-10`, so density residual is not the cause.
- Stable shared linear three-/four-body kernels stop at median/max `5.69%/13.57%` even with a
  numerically pathological `3.55e11` coefficient norm.
- A float64 shared Softplus scalar descriptor residual is implemented and autograd-verified. On the
  original set it reaches per-parent `0.71%--5.08%` with micro-Hartree energy errors.
- `0050129` and deterministic replacement `0033410` are excluded from the formal capacity gate:
  their pre-fit `||H_asym||F/||H_PBE||F` values are `3.68%` and `3.87%`, above the 0.5% numerical
  self-consistency threshold.
- stable5-v2 replaces the unstable parent with `0121249`. Its baseline curl normalized by the PBE
  Hessian is `8.17e-6`; all five frozen parents pass the pre-fit 0.5% numerical gate.
- The shared h128 Softplus scalar residual passed Stage 1 in job 3275. Per-parent complete-total
  relaxed Hessian relative Frobenius is `0.205%`, `0.333%`, `4.889%`, `1.047%`, and `1.118%`;
  median/max are `1.047%/4.889%`. Energy anchor errors are `1.4e-5--1.8e-5 Ha`. Four force MAEs
  are at most `1.04e-4 Ha/Bohr`; hard parent `0132419` is `2.53e-3 Ha/Bohr` and remains an explicit
  tradeoff check.
- Train-parent vibrational diagnostics from the same scalar Hessians give mean frequency
  MAE/RMSE `17.47/46.86 cm-1`, mean mode overlap `0.896`, and 10 predicted versus 11 PBE imaginary
  modes. These are capacity metrics, not held-out generalization.
- Stage 2 is active. The frozen 31-candidate/direction manifest is
  `/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/candidate_direction_manifest.json`,
  SHA256 `8184ae0713f915c69d6614c3e84ce8606d4ea72be0ef6c98f544f66e5f25f403`.
  It selects the first 20 strict curl-stable train parents and fixes train/held-out internal
  directions before fitting. Test100 and validation parents remain unread.
- Baseline array 3297 plus local-checkpoint rescue arrays 3319/3327 are complete. Selection v2
  chose 20 parents with baseline Hessian relative Frobenius median `2.695` and range
  `1.288--3.568`. Each has 17 train and 7 held-out orthonormal internal directions; maximum
  direction nonorthogonality and external-mode overlap are `6.63e-15` and `1.62e-15`.
- Nine candidates fail the PBE-normalized curl gate despite density gradients near `1e-10`:
  `0043438`, `0050129`, `0061724`, `0024609`, `0052472`, `0057862`, `0033410`, `0083824`,
  `0053662`, and `0070824` also fails, for ten total numerical exclusions. Several have curl
  ratios from order 1 to order 1000 and must never enter physical Hessian averages. `0080472`
  passes but is not selected because the first 20 stable candidates were already filled.
- Authoritative selection:
  `/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/selection_v2/selected_baseline_manifest.json`.
  Design-only shared descriptor job 3336 completed in `2:31:28` with `7.22 GiB` maximum RSS. It
  uses zero linear initialization and does not solve coefficients against train, held-out, or full
  Hessian targets. The canceled partial design path is preserved separately.
- The 20-parent job 3337 smoke passed. The formal lambda-H=1 warm-up job 3338 completed 5000 steps
  in `633.9 s` with `11.63 GiB` maximum RSS. It reduced median train-direction, held-out-direction,
  and full-Hessian relative Frobenius from `2.591/2.615/2.695` to
  `0.279/0.571/0.403`. Maximum values are `0.431/1.004/0.564`. This is substantial progress but
  fails all Stage-2 accuracy gates and therefore cannot advance to train100 or Test100.
- Weighted curvature gradients at the end of the warm-up are smaller than the energy gradient and
  PCGrad conflicts are intermittent, not persistent. Job 3339 continues from
  `mlp_h128_warm_v2/best.ckpt` for 30,000 steps with lambda-H=10 and learning rate `3e-5`, while
  retaining E/F replay, dynamic train-direction sampling, spectrum loss, clipping, and PCGrad.
  Its output is `stage2_direction_v1/mlp_h128_h10_v2`. Test100 access remains exactly zero.
- A pre-fit identifiability audit shows that v1 train directions cover only a median `37.8%` of
  each parent's internal-coordinate dimension. The PBE `Q H Q` block not identified by exact v1
  train HVPs has median relative Frobenius `68.6%`. This establishes that the direction protocol
  is globally underdetermined, but it is not a per-parent error explanation: Pearson correlation
  between unidentified `Q H Q` and warm full error is only `0.10`, and the minimum-norm oracle
  held-out error correlation is `-0.08`. Functional representation remains a separate bottleneck.
- Dense-direction Stage-2 v2 was therefore preregistered before fitting with a new seed, at most
  48 internal directions, a 20% immutable held-out split, and up to 8 structured directions per
  kind. Its manifest SHA256 is
  `dfd838b375ae1165708c66cf25a25beb51a3c0c56170e44e416adf1b566134ee`.
  On the same frozen 20 train parents, median direction coverage is `76.9%` and median unidentified
  `Q H Q` relative Frobenius is `41.6%`; orthogonality/external residuals remain about `1e-11`.
  Job 3340 trains this v2 bank with eight dynamically sampled directions per parent. It starts
  from the v1 warm checkpoint, which consumed only v1 train directions and no v2 held-out labels.
  Both jobs run only on node01; Test100 access remains zero.
- Sparse-v1 job 3339 is complete and fails the preregistered gate. At 30,000 steps its median
  train/held-out/full relative errors are `2.23%/39.28%/24.92%`; P80/P90 full errors are
  `36.86%/44.16%`, only `10%` of parents are at or below 15%, and max held-out/full errors are
  `1.403/0.564`. Median energy absolute error and force MAE are
  `5.16e-4 Ha/4.59e-4 Ha/Bohr`, so the failure is unseen-curvature generalization rather than an
  E/F replay collapse. Formal analysis is in `stage2_direction_v1/analysis_final_v1`.
- A strictly isolated 20-parent full-Hessian capacity oracle (job 3341) reached median/max full
  errors `14.2%/23.7%` by step 1400 and formally passed all 20 parents at step 30,000:
  median/P90/max `1.62%/3.99%/4.95%`. Median energy/force errors are
  `2.03e-5 Ha/1.30e-4 Ha/Bohr`. Its capacity-only vibrational metrics are frequency MAE/RMSE
  `20.48/44.16 cm-1`, mode overlap `0.860`, and 42 versus 33 PBE imaginary modes. It is not a
  candidate and its checkpoint cannot initialize a direction-generalization run.
- Near-full v3 was frozen with a new seed after rejecting and preserving two audit failures: a
  single-pass Gram-Schmidt manifest with `3e-7` residual, then job 3342/its manifest because
  structured directions could fill small internal spaces without random-internal directions.
  The generator now uses double reorthogonalization and reserves at least 20% random-internal
  directions. The accepted manifest SHA256 is
  `daf5c2d34a7e83c530852c5989b07aef33fa99943828f5d2718c1627f1cddc2b`; every selected parent has
  8--12 random directions, median train coverage is `89.7%`, median/P90/max unidentified `Q H Q`
  is `24.4%/30.4%/30.4%`, and max orthogonality/external residuals are
  `7.8e-16/1.7e-16`. Job 3343 is the accepted v3 run and starts only from the v1 warm checkpoint.
- Dense-v2 job 3340 is complete. It passes only the median full-Hessian gate: train/held/full
  medians are `3.03%/32.06%/14.73%`, P80/P90 full are `23.73%/30.14%`, and fractions at or below
  10%/15%/20% are `15%/50%/70%`. Frequency MAE/RMSE improves to
  `72.91/188.17 cm-1` with mode overlap `0.747`, but it has 59 versus 33 PBE imaginary modes.
  Sparse-v1 is `213.69/457.53 cm-1`, overlap `0.605`, and 111 imaginary modes. Thus direction
  coverage improves independent curvature and vibration metrics but v2 remains non-promotable.
- Accepted v3 job 3343 completed 30,000 steps. Median train/held/full errors are
  `2.25%/15.05%/5.05%`; full mean/P80/P90/max are `6.52%/10.86%/11.30%/18.83%`, and
  75%/95%/100% of parents are at or below 10%/15%/20%. Thus the formal full-Hessian distribution
  target passes. The stricter direction gate fails: only 80% of parents have train HVP below 5%,
  only 50% have held-out HVP below 15%, and held-out max is 57.5%. Frequency MAE/RMSE is
  `31.73/67.12 cm-1`, mode overlap `0.819`, with 46 versus 33 PBE imaginary modes. Median
  energy/force errors are `3.33e-4 Ha/4.18e-4 Ha/Bohr`.
- The largest held-out errors are not solely small-reference artifacts. Hard parents
  `0038511`, `0064547`, `0132608`, `0031108`, and `0031012` contain real bond/angle absolute HVP
  errors; `0038511` has held/full relative errors `0.575/0.188`. Stage 3 remains blocked by the
  preregistered unseen-direction gate even though the full-Hessian distribution target passes.
- Job 3344 is a same-data/same-seed lambda-spectrum `1.0` ablation because the main run's weighted
  spectrum gradient was negligible compared with HVP. Job 3345 is a 10k, LR `1e-5` continuation
  from the main best checkpoint with mean plus top-20%-train-parent HVP loss. It uses only train
  directions; held-out directions do not affect weights, gradients, checkpoint selection, or
  stopping. Neither run accesses Test100.
- Both diagnostics are complete and neither replaces the accepted v3 model. Spectrum weight 1.0
  gives full median/P90/max `5.05%/11.28%/18.89%`, but worsens median energy error to
  `9.72e-4 Ha`. The top-20% tail continuation improves train/full median to `2.11%/4.74%` and full
  max to `18.02%`, but worsens the original held-out median to `15.72%`; its energy/force medians
  are `4.55e-4 Ha/4.07e-4 Ha/Bohr`. These are non-promoted diagnostics.
- After every model and selection rule was fixed, an independent no-training confirmation bank
  was preregistered. It contains 480 exactly new internal directions, 24 per parent, including
  eight random directions per parent. Manifest SHA256 is
  `ce2e9d026a12c73f1bb1d14986017b89a305cdc781082bac4ceddf38b5ef909d`; maximum overlap with the
  training bank is `0.9709`, and no exact duplicate remains. The main v3 checkpoint passes the
  distribution gate on this bank: parent median/P80/P90/max are
  `5.17%/11.87%/11.97%/18.62%`, with `75%/95%/100%` of parents at or below
  `10%/15%/20%`. Mean component HVP MAE/RMSE are `0.002552/0.005296 Ha/Bohr2`.
- The Stage-2 decision is therefore deliberately two-part. The formal full-Hessian distribution
  target and an independent unseen-direction distribution confirmation pass, so Stage 3 may begin.
  The original all-parent gate still fails because only 50% of parents have the original held-out
  relative error at or below 15%; this caveat remains mandatory and the tail/spec diagnostics are
  not promoted.
- Stage-3 asset audit is in progress. The frozen train100 branch audit contains 100 selected
  train-only parents and 400 directions, but only 31 parents/202 directions pass the hierarchical
  v3 stability protocol and only seven parents pass all four v2 directions. The complete-total
  baseline curl audit leaves 21 physically usable train parents (20 selected plus reserve
  `0080472`). The independent validation20 branch audit has seven v3-stable parents. Existing
  120 analytic PBE Hessians and train800 E/F labels are reusable; historical direct-HVP sidecars
  are fixed-density learned-component surrogates and must not be relabeled as complete-total
  relaxed model HVPs. Test100 access remains zero.
- **2026-07-21 provenance correction:** the authoritative Stage-2 manifest was found to mix two
  frozen base checkpoints. Seven parents (`0016298`, `0028399`, `0064547`, `0031108`,
  `0132419`, `0031012`, `0121249`) use a checkpoint capacity-fitted on `0028399`; the other 13
  use the original frozen A checkpoint. The shared geometry residual is conservative, but adding
  it to parent-dependent baselines does not define one deployable scalar total energy. Therefore
  all Stage-2 v1/v2/v3 and independent-confirmation numbers remain useful representation and
  direction-coverage diagnostics, but the previous formal Stage-2 pass is revoked until a
  uniform-baseline rerun reproduces it.
- Stage-3 `assets_v1` is rejected because it omitted seven known baseline records; `assets_v2`
  repairs that merge and freezes 3200 train800 replay labels, 21 mixed-baseline eligible HVP
  parents, and seven validation candidates, but is now also non-authoritative for model training
  because of the heterogeneous baseline provenance. Both directories are preserved for audit.
- Replay job 3370 used the capacity-fitted checkpoint and was stopped as soon as this provenance
  issue was confirmed; its partial output is preserved as rejected. Uniform original-A jobs are
  3379 (seven missing 20-parent full baselines) and 3380 (16-point strict E/F replay smoke).
  Test100 remains untouched.
- Uniform original-A replay smoke 3380 completed in `6:32` with `15/16` strict successes. On the
  successful points, median wall time was `22.99 s`, median cycles `766`, median tensor projected
  density-gradient norm `1.57e-10`, and maximum tensor-vs-legacy scalar-energy disagreement was
  `6.82e-13 Ha`. This verifies the scalar replay baseline path for ordinary geometries.
- The sole failure is `0000023.0000000`: the legacy optimizer reports `1.02e-10`, but rebuilding
  the same final density in the differentiable complete-total geometry path gives projected norm
  `0.478` and a `0.020276 Ha` energy mismatch. It is a reproducible tensor/legacy stationarity
  mismatch, not an insufficient-cycle failure, and must be excluded or repaired before replay
  supervision is treated as complete.
- Focused rebuild audits 3394/3395 show that an ordinary fixed-geometry rebuild also fails to
  reproduce the optimizer state (`0.31--0.43` projected norm and `0.0087--0.0175 Ha` legacy-energy
  mismatch). The basis matrices themselves are numerically invertible: density coefficient
  round-trip relative error is `1.64e-14`, and both matrix identity errors are below `6.4e-14`.
  Independent rebuild energies vary for this symmetric geometry, pointing to a non-unique local
  frame/graph reconstruction branch rather than optimizer cycles or matrix inversion. Keep this
  parent behind the derivative-stability mask pending a deterministic-frame audit.
- Initial array 3379 passed only the first molecule because a comma-delimited Slurm `--export`
  value was parsed as separate environment entries. No scientific task failed. Job 3387 resubmits
  the remaining six IDs with a colon-delimited list; jobs 3379_0 and 3387_0--5 now compute all
  seven uniform original-A baselines on node01.
- The first completed uniform baseline, `0016298`, has original-A relative Frobenius `2.3370`,
  versus `2.2165` in the heterogeneous manifest. Its curl ratio remains small (`6.21e-5`), so the
  parent stays numerically admissible, but this single comparison is not a Stage-2 rerun result.
- `qm9_complete_total_replay_baseline.py` now has an opt-in
  `--audit-fixed-geometry-rebuild` consistency diagnostic. Its replay/merge tests pass `4/4` on
  node01. Full uniform-A replay array 3396 uses the frozen 3200-label train800 CSV and starts at a
  two-GPU throttle while six Hessian baselines occupy the other GPUs; it does not access Test100.
- All seven missing baselines are complete. A new parameter-level provenance audit compares every
  embedded step-0 state tensor with frozen original A, allowing only exact-value float32-to-float64
  promotion. The seven-run audit and the audit of all 28 complete `baseline_runs` records pass
  `7/7` and `28/28`. Frozen original-A checkpoint SHA256 is
  `e6516b04917a9dfaae3c4d960b10d77c768f3e5e784f95290ff41c1202f9d9bc`; canonical parameter
  hash is `5eae47caa18ba8a6e7e96c13e0e1c4418ba4106bf4d4b5385d09e75b9d880653`.
- Uniform selection is frozen at
  `stage2_direction_v1/selection_uniform_A_v1/selected_baseline_manifest.json`, SHA256
  `e81e00ecbf99e7d14751c079aeb081d06455ca720bbb3ca5c758e39de359f127`. It selects the same 20
  parents in the same order as v2, with 21 total numerical passes. Baseline Hessian relative-Fro
  median changes from `2.6953` to `2.7331`; energy/force medians are
  `0.09216 Ha/0.10181 Ha/Bohr`.
- The 9.1-GiB descriptor derivative design is reusable because selected parent geometry/order is
  unchanged and the cached arrays contain scalar descriptor values/derivatives and PBE-only
  normalization, not fitted baseline targets. New MLP summaries now record the baseline manifest
  hash, design summary/feature-key hashes and matrix sizes.
- Uniform training chain is active: job 3402 performs the matched 5000-step v1 warm-up from zero;
  dependent job 3403 starts near-full v3 from that uniform warm checkpoint for 30,000 steps. No
  mixed-baseline checkpoint initializes either run.
- Replay slowdown was traced to a separate non-Slurm `mpn_system_diag` workload occupying most
  physical CPU cores. That user workload completed at `05:26`; replay array 3396 was retained and
  concurrency raised to seven alongside the one-GPU MLP job. Existing successful points resume
  by task hash and are not recomputed.
- Uniform warm-up job 3402 completed 5000 steps from zero. At best step 4300 its full-Hessian
  median/max relative Frobenius is `0.3717/0.7828`, with train/held-out direction medians
  `0.2328/0.5188`. This is a warm start, not a promotable result.
- Uniform near-full v3 job 3403 completed 30,000 steps in `42:43`; its frozen best checkpoint is
  step 30,000, SHA256 `bc0cb2e63aad323ab011a92d806ce85f44475325b79aebabaab19c989026f6ea`.
  Final train/held-out HVP medians are `0.02340/0.15668`. Full-Hessian
  median/mean/P80/P90/max are `0.05431/0.06444/0.08151/0.12125/0.16402`, with
  `85%/90%/100%` of parents at or below `10%/15%/20%`. Median energy absolute error and force MAE
  are `4.55e-4 Ha/4.02e-4 Ha/Bohr`; maximum asymmetry-to-symmetry ratio is `5.60e-4`.
- Frozen audit job 3408 completed. The independent 480-direction confirmation passes its
  distribution gate: parent median/P80/P90/max are `0.05247/0.09823/0.12447/0.16264`, and
  `80%/95%/100%` are at or below `10%/15%/20%`. Component HVP MAE/RMSE are
  `0.00248/0.00515 Ha/Bohr2`; frequency MAE/RMSE are `29.85/63.13 cm-1`, mean mode overlap is
  `0.821`, and model/PBE imaginary totals are `41/33`.
- Stage 2 therefore passes the full-Hessian and independent-direction distribution criteria, but
  still fails the preregistered strict all-parent direction criterion: only `90%/45%` of parents
  pass train <=5% / held-out <=15%, and held-out max is `0.7215`. Stage 3 may proceed with this
  mandatory tail-risk caveat; this is not an all-parent direction-generalization pass.
- Authoritative Stage-3 pretraining assets are now
  `stage3_unseen_parent_v1/assets_v3_uniform_A`. The asset manifest SHA256 is
  `4eb176bde25d06c4ad1480bbe2b19752ff7cca23d0a3a768b36d3ed55ad3c703`; its new independent
  validation capacity manifest SHA256 is
  `95c9c2a7585ed3508bc27ee9b749a8201868050aebfaf5a50bdfcfa4baff911f`. It freezes seven
  validation-only parents and certifies zero Test100 access.
- Uniform train800 replay array 3396 had written 683 resumable task summaries by `06:03`; one
  known symmetric-frame failure (`0000023.0000000`) remains masked. Its launch throttle was
  temporarily lowered to one without interrupting running tasks so seven validation baselines can
  use freed node01 GPUs. Job 3416 restores replay throttle seven after validation job 3415 ends.
- Validation baseline array 3415 evaluates original A on all seven frozen validation parents.
  Job 3420 then requires exact checkpoint provenance, strict density gradient below `1e-8`,
  PBE-normalized curl below `0.5%`, and 7/7 complete full Hessians before emitting the validation
  baseline manifest. Failure produces an audit CSV and blocks Stage 3. No Test100 path is present.
- Active-feature schema job 3426 completed in `1:04`. The frozen schema binds the step-30,000
  checkpoint to 20,328 active scalar features: `7,992/10,095` three-body and
  `12,336/12,336` four-body. The manifest captures checkpoint/design/key/schema hashes and records
  zero Test100 access. Exact-collinear four-body torsions now use a second-order-safe branch;
  combined Hessian tests pass `5/5` while the nondegenerate formula is unchanged.
- Replay descriptor caching and independent-parent external evaluation are implemented and tested.
  The cache stores float64 descriptor/Jacobian arrays plus residual energy/force targets and is
  keyed to replay-baseline and active-schema hashes. It remains unlaunched pending replay merge
  job 3422 and a one-task smoke. The external evaluator can consume only frozen active keys and
  reports coverage; it has no Test100 input.
- Validation baseline merge is complete for all seven frozen unseen parents. Its manifest is
  `stage3_unseen_parent_v1/validation_baselines_v1_uniform_A/validation_baseline_manifest.json`,
  SHA256 `b1937e31c13e9906b94e8bdf1516e988adad974fc70dc34eb5292eeae108772d`.
  All `7/7` runs are strict, with maximum density-gradient norm `1.4424e-10` and maximum
  PBE-normalized curl `8.526e-4`; untouched original-A Hessian relative-Frobenius median is
  `2.7031`.
- The Stage-2 step-30,000 checkpoint fails the independent-parent gate. On the same seven parents,
  Hessian relative-Frobenius median/mean/P90/max are
  `303.611/11642/32703/75040`, energy absolute-error median is `32.215 Ha`, and force-MAE median
  is `13.2666 Ha/Bohr`; no parent is below `20%`. Frequency MAE/RMSE are
  `45278.77/71787.69 cm-1`, mean mode overlap is `0.5788`, and model/PBE imaginary totals are
  `117/12`. These are validation-only results; Test100 remains unread.
- External component diagnostics show that numerical symmetry is sound (maximum asym/sym
  `4.34e-5`) but descriptor scale is not. Four hard validation parents have descriptor L2 norms
  from `1.72e3` to `1.03e6`; the largest descriptor component is `5.41e5`. Linear-only and
  atom-extensive reinterpretations still fail. The root cause is division by tiny active-feature
  column norms estimated on only 20 parents, compounded by incomplete unseen-chemistry feature
  coverage. The old active-schema cache/checkpoint is therefore diagnostic only and may not seed a
  formal Stage-3 model.
- Repair jobs 3448 and 3449 are running on node01. They compare atom-extensive scalar energy under
  the old column scale against atom-extensive unit physical scaling, respectively, using only the
  stable5 Stage-1 parents. Both must recover the `<=5%` capacity ceiling before a new Stage-2 run.
  Formal train800 replay fitting waits for this gate; no candidate is allowed to advance on the
  basis of the old scaling.
- Replay-label array 3396 remains resumable and had produced `1299/3200` geometry summaries at
  08:14. Concurrency is intentionally two because seven nested-thread shards oversubscribed the 76
  physical CPUs and reduced aggregate throughput. The known symmetric-frame failure
  `0000023.0000000` remains masked; no existing successful record is recomputed.
- Feature preconditioning now supports `floored_column_norm`, defined as division by
  `max(source_column_norm, floor)`. Focused tests pass `9/9` locally and on node01. The source
  norm ranges are `9.49e-7--93.48` for active three-body features and
  `1.61e-3--1.41e4` for four-body features, explaining why both unconstrained inverse-column
  scaling and pure unit scaling are problematic. Jobs 3452--3457 scan extensive stable5 floors
  `0.01/0.1/1.0`; jobs 3458--3463 are dependency-gated matched non-extensive controls.
- Scale selection is frozen before those runs complete: choose the largest floor that passes all
  stable5 parents at `<=5%` Hessian relative Frobenius without material E/F regression. The seven
  unseen validation parents may not choose the floor. If no floor passes, move to a higher-capacity
  or local-additive conservative scalar instead of weakening the gate.
- The extensive floored scan is complete and none passes. Floors `0.01/0.1/1.0` give stable5
  median/max Hessian errors `4.18%/8.70%`, `4.50%/9.57%`, and `5.19%/10.35%`. Before the pending
  non-extensive scan completes, E/F preservation is made explicit: stable5 energy median/max must
  be `<=1e-3/2e-3 Ha`, and force-MAE median/max must be
  `<=1e-3/3e-3 Ha/Bohr`. The extensive candidates fail and are diagnostic only.
- All three matched non-extensive floored candidates pass. The preregistered largest-floor rule
  selects floor `1.0`: stable5 Hessian median/max `1.185%/4.925%`, energy median/max
  `6.34e-5/8.75e-5 Ha`, and force median/max `8.70e-5/2.534e-3 Ha/Bohr`. Selection manifest
  SHA256 is `87b27c0e11aefaf80374b349176281fc0edf1e95f0798d11b97187e8b21aa972`; it uses no
  validation-parent or Test100 metrics.
- Stage2-design-complete provisional assets exist. The cutoff-zero schema contains `22,431` keys,
  split `10,095` three-body and `12,336` four-body, with floor `1.0`. Its manifest SHA256 is
  `10ab6938afe2cec659aaa0caf06eb0094e1a4e29d2d81bffaf887c0b9437e9c9`; the zero-residual
  dimension seed checkpoint SHA256 is
  `3d4e5c9f902730914f6e40ab2ef2feb8a7559342ee766f304ffa279f6e9fcb76`. The step-0 seed is
  not a scientific candidate. A follow-up coverage audit showed that the 20-parent design key
  union is still smaller than the train800 chemistry key union, so these assets are diagnostic and
  may not drive replay caching.
- Frozen failure diagnostics now include a reproducible correlation CSV/JSON and PNG in
  `validation_external_eval_stage2_s30000_uniform_A_diagnostic_v2/scaling_analysis_v1`.
  Descriptor-L2, descriptor-max, and correction-Hessian norms have log-error Pearson correlations
  `0.956/0.962/0.995`, providing quantitative evidence for the scaling diagnosis.
- Pending cache jobs 3470/3471 were canceled before they ran; no artifact was lost. A new
  hash-bound inventory scans all 3200 train800 geometries and unions their keys with Stage2 keys.
  Inventory v1 is rejected for mismatched four-body defaults (`0.4/3/1.25`); the corrected
  definition is `sigma=0.75`, torsion order `5`, bond scale `1.35`, exactly matching design and
  cache code. Validation and Test100 remain excluded.
- The authoritative train800-union inventory v3 is complete and supersedes the provisional
  22,431-feature schema for replay work. It contains `32,121` keys (`11,625` three-body and
  `20,496` four-body), adding `1,530/8,160` train800 keys without losing a Stage2 key. Inventory
  manifest SHA256 is `42ab2a5393ee55598d562d299f958297faf2b582b38d2e6f1758caa2fe02ec6a`.
- Jobs 3477/3478 completed a zero-residual exact-dimension checkpoint and schema binding. That v1
  path is retained only for dimensional diagnosis because it discarded the passed stable5 scalar.
  Its schema manifest is
  `stage3_unseen_parent_v1/active_feature_schema_train800_union_floor1_v1/active_feature_schema_manifest.json`,
  SHA256 `5f0c58413d0437e4613b7526efda1f6ea0ad80ff668f974a25f1611ad715093b`; its checkpoint SHA256
  is `256a97f92be6edfd3e580504464747c7a0b35b9a803a712cdd5769246ef8ad81`. Both certify zero
  Test100 access. Runtime descriptor settings are checked exactly against every active schema.
- Cache jobs 3479/3480 were canceled before execution. Jobs 3484--3486 instead freeze the selected
  stable5 checkpoint's 17,765-feature source schema and embed it into the 32,121-feature inventory
  with exact key mapping and scale conversion; new columns are zero. The expanded checkpoint
  SHA256 is `4ab80b076968d7b1f34ac83df61809de1000574f90dafadc1574e8322867f76b` and authoritative v2
  schema SHA256 is `79d05c1351fb48532aa7569af3123b115e69c631d73dc00ae667bfa4c9870e67`.
- Equivalence job 3487 exactly reproduces the stable5 scientific metrics after expansion:
  Hessian median/max `1.1853%/4.9251%`, with energy, force, and Hessian gates all passing. Focused
  node01 checks pass `17/17`.
- Train800 replay and the authoritative descriptor cache are complete. Replay has `3199/3200`
  strict geometry records over all 800 parents; only deterministic bad point `0000023.sample0`
  is excluded. The cache manifest SHA256 is
  `322a4597d7afcd89bdce8b7aec7bbbceaa8c42a198c63810d857786708455489`, contains all 3199
  records, and is bound to schema SHA256
  `79d05c1351fb48532aa7569af3123b115e69c631d73dc00ae667bfa4c9870e67`.
- Preflight passes exact hashes, 32,121 dimensions, 800 replay parents, HVP-parent inclusion,
  validation disjointness and zero Test100 access. A 10-step smoke passed replay loading, complete
  scalar E/F/Hessian forward, PCGrad backward, logging and checkpoint persistence.
- The smoke diagnosed curvature-gradient domination even when task cosine was positive. The shared
  PCGrad utility now optionally caps curvature gradient norm relative to replay gradient norm before
  conflict projection and records both the scale and balanced ratio. Focused node01 tests pass
  `26/26`.
- Replay-aware Stage-2 pilot job 3624 completed 5000 steps at `lambda_H=1`, Adam `3e-5`, warm-up
  1000 and curvature/replay gradient-ratio cap `1.0`. Full-Hessian relative-Frobenius median/max
  improved `2.259/54.572 -> 1.102/2.230`; training-direction median/max improved
  `2.221/55.291 -> 1.084/2.273`; held-out-direction median/max improved
  `2.197/68.085 -> 1.146/1.437`. No parent is yet below `0.15`, and 20-parent energy/force
  median errors are `6.77e-3 Ha` and `1.45e-2 Ha/Bohr`, so every scientific advancement gate
  remains false. Asym/sym median/max are numerically sound at `1.51e-5/6.10e-4`.
- Job 3624 took 20:58, 13.4 GiB maximum host RSS and 40.4 GiB peak GPU memory. Best checkpoint
  SHA256 is `7da7b481f2278b4c6732e68249d7693c28007938392518881689b5867cd183a0`. The output directory
  says `h10` for historical naming reasons, but the saved config records the actual `lambda_H=1`.
  The launcher has a `continue-pilot` profile that resumes this checkpoint and Adam state rather
  than silently resetting the optimizer.
- Continuation job 3625 completed 25,000 additional same-objective steps. Fixed replay-aware
  selection chose continuation step 23,800, cumulative step 28,800. Selected full/train/held-out
  relative-Frobenius median/P90/max are respectively
  `0.566/0.814/0.935`, `0.549/0.830/0.938`, and `0.679/0.867/1.042`; zero of 20 parents is below
  even `0.20`. Full/train improve on `20/20` parents and held-out on `19/20`, but all absolute
  advancement gates remain false.
- Selected 20-parent energy-error median/max are `3.28e-3/1.64e-2 Ha`, and force-MAE median/max
  are `3.88e-3/9.66e-3 Ha/Bohr`. Numerical symmetry remains sound at asym/sym median/max
  `1.51e-5/6.12e-4`. Held-out direction-family medians are `0.666` bond, `0.759` angle,
  `0.741` torsion, `0.630` random-internal, and `0.587` low-mode; no direction family approaches
  `0.15`. Hard parents are `0016142`, `0132608`, `0049017`, and `0038511`, with `0132608` the
  held-out maximum and only held-out regression from s5000.
- Job 3625 took 1:38:37, 13.4 GiB maximum host RSS and 40.4 GiB peak GPU memory. Best checkpoint
  SHA256 is `83ac62943deb4fbc05cbe01428222c68a8b213c0c1f50df5eb9648e7e75b9b79`; summary SHA256 is
  `68ec2dffeaa044428003aca9ae2d1a4cad43557b21566d67b4cbe80760a657f2`. The robust 30k result
  is not promoted. Validation parents and Test100 remain unread.
- The no-replay control plumbing now applies HVP warm-up independently of replay. Job 3627 was
  stopped and preserved because an inherited curvature/replay norm cap used a vanishing E/F task
  norm and suppressed HVP by about `3e-4`; it is not a scientific comparison. Job 3628 verified
  the corrected uncapped path, and formal job 3629 completed 30,000 steps in `45:08` with no NaN
  or OOM.
- Job 3629 selects step 29,400. Full/train/held-out relative-Frobenius median/P90/max are
  `0.255/0.399/0.455`, `0.240/0.379/0.423`, and `0.410/0.688/0.906`. It beats the replay-aware
  result on all `20/20` parents for every curvature aggregate; paired median ratios are
  `0.431/0.409/0.658`. Energy-error median/max are `6.18e-4/3.01e-3 Ha`, force-MAE median/max
  are `1.02e-3/3.17e-3 Ha/Bohr`, and force improves on `20/20` parents. Nevertheless only
  `2/20` full Hessians are below 15%, no held-out parent is below 15%, and all Stage-2 gates remain
  false.
- The no-replay checkpoint SHA256 is
  `e0bf555eac59cd6f66fcd8ff63a61be82ff39a1634524636dcd6413aa5316ca2`; summary SHA256 is
  `416262ee2fdaa5008cd9335299fead48366310b48b34982c90cedec7532f7c4a`. This establishes both
  replay optimization drag and a remaining global-h128 capacity/generalization defect. The next
  Stage-2 experiment is an exact function-preserving h128-to-h512 widening on the identical
  no-replay split; only after a capacity pass may replay be reintroduced. Validation and Test100
  remain unread.
- Function-preserving width expansion is implemented in
  `scripts/expand_qm9_complete_total_hidden_width.py`. It copies h128 exactly, adds 384 seeded
  incoming rows with zero output weights, drops incompatible Adam state, and refuses Test100-
  exposed sources. Its tests plus the existing scalar tests pass `12/12` on node01.
- The h512 initial checkpoint SHA256 is
  `e5fb4c632cccc79036510872de93ae8488c9bec6db11b38367e5910420ee9d8b`. Smoke job 3630
  exactly reproduces h128 step-0 full/train/held medians `0.25515/0.24021/0.40983`, remains finite
  for 10 steps, uses 13.4 GiB host RSS and 41.22 GiB peak GPU memory, and records zero Test100
  access. Formal no-replay h512 job 3631 is active for 30,000 steps on node01; do not infer a
  capacity pass from the smoke.
- A matched h128 restart is preregistered before h512 completion so width is not confounded with
  30,000 additional optimizer steps. Both arms start from the same selected step-29,400 h128
  scalar, discard Adam state, reset the same random-direction sequence, warm up HVP for 1000 steps
  and run 30,000 steps. Only the 384 initially dormant h512 units differ.
- Matched h128 job 3632 is active alongside h512 job 3631. The new read-only comparison utility
  `scripts/qm9_complete_total_compare_geometry_mlp_runs.py` passes `2/2` focused tests, requires
  identical parent sets and frozen-Test100 certificates, and writes hashed run provenance,
  per-parent/pairwise CSVs, distributions and trajectory plots.
- Dormant h512 still barely activates its new units by step 4000, so a preregistered paired-active
  width control uses 192 identical-input pairs with exact `+1e-3/-1e-3` output cancellation.
  This preserves initial scalar E/F/Hessian while giving nonzero opposite incoming-weight
  gradients. Expansion tests pass `3/3`; checkpoint SHA256 is
  `2453fdff695be3ab470709f592294bbb26fa6803e8b1bb695ac1343050cfff24`.
- Paired smoke job 3633 is finite for 10 steps and gives full/train/held medians
  `0.25518/0.23878/0.40802`, 13.4 GiB host RSS, 41.22 GiB GPU peak and zero Test100 access.
  Formal paired-h512 job 3634 is active with h128 job 3632 and dormant-h512 job 3631. All three
  share source function/data/steps/LR/warm-up/seed; only hidden width/initialization differs.
- A depth-two scalar branch is implemented with analytic chain-rule E/F/Hessian and no descriptor-
  space Hessian materialization. The scalar remains the sole force/Hessian owner. Float64 autograd
  and function-preserving expansion tests pass `13/13`. Initial deep h128x256 checkpoint SHA256 is
  `37adf84df97fb1b37033fed3e61502dd5ee4a92b79d5d2df3e6004dc64cfc950`.
- Deep smoke job 3635 finishes 10 finite steps at full/train/held
  `0.25518/0.23879/0.40838`, 13.38 GiB host RSS and 40.28 GiB GPU peak; summary SHA256 is
  `d5d4e80bf5e05dfd6cd921b2fab48e6887b0cf4ece32cf5ef471d7a5e4beef10`. Formal no-replay
  deep job 3636 is active because matched h128/h512 remained far from the Stage-2 gate. It is a
  structural capacity control and does not authorize replay, validation or Test100.
- Jobs 3631/3632/3634/3636 are now complete. Matched 30k full/train/held medians are h128 restart
  `0.18370/0.16555/0.33014`, dormant h512 `0.14975/0.12818/0.29778`, paired h512
  `0.18186/0.16529/0.32950`, and deep h128x256 `0.18926/0.17122/0.33883`. No arm passes the
  Stage-2 train/held gates. Dormant h512 is retained only as the best train-only source.
- Dormant h512 lowers frequency MAE/RMSE to `129.91/231.86 cm-1` and mode-overlap mean to
  `0.62885`, but predicts 90 imaginary modes versus PBE's 33. Numerical asymmetry is already below
  the 0.5% gate, so numerical conservation is not the remaining blocker.
- The frozen comparison, direction, vibrational and activation artifacts are under
  `/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/stage2_width_depth_comparison_v1`.
  Its summary SHA256 is `23a6cea4c3fa9f0fca0a5c6ac382363c454fc74d6a7c9c05d95c041093a51f80`.
- The active replacement is a local-additive conservative scalar in
  `mldft/ml/models/components/local_body_order_residual.py`, trained by
  `scripts/qm9_complete_total_local_body_order_capacity.py`. It contains typed pair/triplet and
  optional bonded parity-even torsion terms, locked `+a/-a` output pairs, C3 cutoff, float64, and
  no force head. Focused local/trainer tests pass `14/14` on node01.
- Independent canceling outputs were rejected because Adam broke their zero baseline and caused
  large E/F drift. Locked antisymmetric pairs fix that failure. Batch-1 job 3642 and full-batch
  smoke 3643 are preserved negative controls; locked 1000-step job 3645 is stable but changes
  curvature by less than `1e-5`.
- Reference-geometry Taylor anchoring is implemented as
  `delta E(R)=g(R)-g(R0)-grad g(R0).(R-R0)`. At `R0` it exactly preserves source energy/force and
  retains the local Hessian. It is a conservative local curvature correction around a supplied
  reference geometry, not yet a globally transferable energy functional.
- Anchored pair+triplet job 3655 uses all 35--52 frozen train directions per parent and selects
  step 20 at full/train/held `0.14866/0.12797/0.29602`; E/F are unchanged. It improves held
  bond/angle, but regresses held torsion/random/low-mode and improves only 12/20 held-parent
  aggregates. It fails the Stage-2 gate.
- A bonded four-body torsion branch is now under matched control. Uniform LR job 3656 is rejected
  because torsion HVP gradients are about 780 times pair/triplet gradients. Separate parameter
  group job 3657 with pair/triplet LR `1e-4`, torsion LR `1e-6` restores favorable one-step
  full/train/held `0.14944/0.12807/0.29760`. Matched 20-step job 3658 completed at
  `0.148468/0.127900/0.295843`, but the gain over pair+triplet is only `1.75e-4` on held median
  and held P90/max worsen. It is rejected for continuation.
- Convex anchored descriptor regression and shared internal-coordinate quadratic models were then
  evaluated with train-only ridge selection. Their best shared held medians remain
  `0.29628` and `0.29385`; neither approaches the `0.15` gate. A parent-specific full-cross
  oracle reaches full/train/held `0.11091/0.08669/0.27415`, proving train-covered matrix capacity
  while exposing the unresolved held-direction response.
- A shared local-environment coefficient MLP now uses initial-function subtraction: its correction
  is exactly zero at initialization while all hidden layers receive gradients. Focused tests pass
  `6/6`. Corrected jobs 3673/3674 optimize the train-only calibration objective but finish at
  full/train/held `0.15031/0.12891/0.29792` and `0.14969/0.12829/0.29809`. The route is rejected,
  because optimization succeeds without held-direction generalization.
- Final local-architecture analysis is under
  `/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_architecture_final_direction_analysis_v1`.
  Summary SHA256 is `376a15096f1dc0548538d5d4f9b629ac29b0f716737b61432294ff4d2150e8a4`.
  At that checkpoint no local Stage-2 candidate passed and Test100 access remained exactly zero.
- The next architecture is an orthogonally equivariant spectral plus atom-block response operator
  in `mldft/ml/models/components/spectral_hessian_residual.py`. A reference-anchored scalar
  quadratic owns its Hessian while preserving source E/F exactly at `R0`. The uncompleted shared
  block operator reaches full median `0.0912` but misses held tails; chemistry conditioning lowers
  full median to `0.0777` and held median/max to `0.0877/0.1466` but still misses the train maximum.
- Exact symmetric HVP completion in fresh protocol v5 then passes all labeled-parent Stage-2
  gates. The h512-source result has full median/P90/max `0.0124/0.0222/0.0260` and held
  `0.0494/0.0876/0.1179`; the replay-source control has full median/max `0.0142/0.0533` and held
  median/max `0.0548/0.1376`. The operation is label-conditioned and is not available on an unseen
  parent. h512/replay labeled-parent frequency MAE is `8.33/10.89 cm-1`.
- Before external evaluation, protocol
  `configs/audit/qm9_complete_total_hessian_unseen_parent_shared_operator_v1.yaml` froze the two
  arms, seven validation parents, all hashes, no HVP labels/completion, Test100=0, and Hessian plus
  5% E/F regression gates. Protocol SHA256 is
  `c0e2accee7452c68bc54f2fcfd81fb2dec1771be0211e2a2ab73b9e77c954743`.
- Unseen-parent validation decisively fails. h512/replay shared operators have Hessian median/P90
  `8.881/251.613` and `5.140/1515.609`; both have `0/7` below `0.15`. Replay preserves useful E/F
  (`force median/P90 0.0323/0.0679 Ha/Bohr`) but cannot transfer curvature. Frequency MAE is
  `5768.6/6706.0 cm-1`, with 108/63 predicted imaginary modes versus PBE's 12.
- OOD amplification is the diagnosed mechanism. Across 14 arm-parent points, parent-feature
  `max |z|` and log correction scale have Spearman `0.899` (`p=1.24e-5`). `0056113` has z-score
  `41.9/32.2` and correction/PBE Frobenius `572/3871`. Some z<2 parents improve, so the operator
  contains useful structure but its source response and chemistry conditioning are unbounded.
- Paired rows and the diagnostic figure are under
  `stage3_unseen_parent_v1/validation_spectral_block_shared_operator_comparison_v1`; summary and
  plot SHA256 values are `6a541e046844a977b041ab34ad7afe1e622c327025c560a494f66efc79c34591`
  and `27bed3be797e051886638707afdd9cb970c66aefc3420c2e670ed5402c31621b`.
  Stage 3 is rejected; no train100 expansion occurred and Test100 access remains exactly zero.
- The replacement v9 nonlinear local-random-feature scalar passes the fitted-parent stable5
  capacity and vibration gates: Hessian relative-Frobenius median/P90/max is
  `0.01174/0.02413/0.02833`, frequency MAE/RMSE is `8.19/23.40 cm-1`, and mean mode overlap is
  `0.9401`. It is still a fitted-parent ceiling rather than transfer evidence.
- On the frozen train20 split, v9 fits its 17 registered train directions at median `0.00952` but
  fails the seven held directions at median/P90 `1.7439/9.5859`; full-Hessian median/P90 is
  `1.1407/4.5739`. Ridge, random-feature-width, and extra-direction diagnostics improve some
  medians but no arm passes. A full-matrix fit-only control reaches median/P90 `0.06683/0.08732`
  and `20/20 <=0.15`, proving train20 complete-matrix capacity while isolating missing shared
  direction structure.
- Label-independent CountSketch jobs 3833--3839 also fail. The best held-selected 12,288-dimensional
  arm has train/held/full medians `0.07699/0.63630/0.50547`, full P90 `3.65303`, and
  `0/20 <=0.15`; all six dimensions fail. Generic random compression is therefore frozen as a
  negative diagnostic. Summary/aggregate/direction SHA256 values are
  `bb1745c52711d47323f3277ac231549f9653d0f8ffd3716ce96a271e831c2d2c`,
  `baa88be421302001340b8beb34d0a2f543563811f4a7495c7dacc5f9d66e149c`, and
  `905db9dae7ae620c89661107e824188709736e48db1597a50059d96f2359be2c`. The authorized next
  branch is a chemistry-aware scalar coefficient basis tied over element, radial, angular, and
  torsion metadata, still on train20 only.
- Chemistry-aware typed-DCT jobs 3840/3841 also fail the frozen train20 gate. The best exposed-held
  arm `a3_f2_r2` has train/held/full medians `0.2225/0.6147/0.4787`, full P90 `1.1584`, and
  `0/20 <=0.15`; the wider `a3_f3_r2` improves train median to `0.1357` but regresses held median
  to `0.9870`. This branch is a conservative scalar and preserves E/F, but low-order smooth
  coefficient sharing underfits while added radial capacity overfits directions. Analysis summary
  SHA256 is `770ea6958e075b146a1704fa6ef57a603895cdaa1902b8d466d6f95ad22699ed`.
- Stable5 v9 coefficient-prior job 3846 is catastrophic on the 15 train20 parents not used in its
  full-Hessian fit: full-Hessian median/P90 is `170.59/3356.55`, held-HVP median/P90 is
  `118.07/1612.29`, energy median is `6.05 Ha`, and force MAE median is `3.19 Ha/Bohr`. The five
  fitted parents retain their expected 1.17% full median. This clean seen/unseen split proves that
  the v9 coefficient vector is an interpolant, not a transferable prior. Summary SHA256 is
  `38dae5170d5396a5673cc4f0317a1596e942451f0bc9a4f025a2248cddcb0569`; do not fit residuals
  around this prior.
- A bounded scalar/vector equivariant local-energy model is now implemented. Its second-order-safe
  normalization, rotation/translation invariance, force covariance, Hessian symmetry, and Hessian
  parameter gradients pass remote tests (`8/8`). Five-step stable5 smoke job 3847 lowers Hessian
  median `2.72677 -> 2.68149` with exact E/F preservation, `1.15e-16` asymmetry ratio, and about
  1.0 GiB peak GPU allocation. Formal 3,000-step job 3848 is active on one node01 A100; it is still
  a capacity test and authorizes no train20/validation/Test100 advance.
- Stable5-only width/depth smoke job 3849 selects h64/l2 as the sole capacity expansion. At step
  20, h64/l2 gives median/P90/max `2.148/2.538/2.794` in 179 s, while h64/l3 gives
  `2.136/2.523/2.777` in 266 s. The third interaction's 0.6% median gain does not justify 49%
  additional wall time. h32/l2 job 3848 was stopped after its median plateaued around `1.03` by
  step 350; all artifacts were retained. Formal h64/l2 job 3851 is active on node01 and reaches
  median/P90/max `1.022/1.151/1.155` at step 200 with exact E/F anchoring and `3.1e-16` maximum
  asymmetry, but it remains far above the 5% stable5 gate.
- A hash-bound optimizer audit freezes the h64/l2 step-100 checkpoint at
  `/scratch/xzh/models/complete_total_capacity/20260717/stage5_bounded_equivariant_v2/lbfgs_v1/adam_best_through_s100.ckpt`
  (SHA256 `3422fb7f891e31e016e31a8013d14975612e6dc924252cf89bcc59e15746a12a`).
  Full-stable5 joint LBFGS job 3852 runs in parallel on node01 under
  `qm9_complete_total_bounded_equivariant_lbfgs_v1.json`; validation and Test100 remain frozen.
  Passing requires every stable5 parent at or below 5%, not merely a lower median.
- That optimizer audit is now complete and fails: full-stable5 LBFGS gives median/P90/max
  `0.831/1.008/1.045` after 30 iterations, while five independent one-parent LBFGS runs end at
  `0.506--0.933`. All report no parent-CV authorization. The failure is therefore not primarily
  Adam, minibatch noise, or cross-parent conflict; the current `l<=1` response span is inadequate.
- Extending the cutoff from 8 to 20 Bohr with matched radial spacing gives only a 5% 20-step
  improvement (`2.036` versus `2.148`), but the advantage grows in formal training. Latest E4
  step 600 is `0.595/0.702/0.762`, while E2 step 1050 is `0.659/0.772/0.840`.
  Both remain active as same-budget controls. The hash-frozen E4 step-350 LBFGS audit is complete:
  30 iterations lower median/P90/max to `0.574/0.680/0.736`, but still fail the all-parent 5%
  gate. Its summary SHA256 is
  `fd12fd34ffdd3c43a54d6f0ab9d52ee6edc65861c7a6689d13156988eccdc35e`.
  Since the 30-iteration curve was still descending, node01 job 3877 runs a frozen 100-iteration
  extension from immutable Adam step 750. Its checkpoint SHA256 is
  `2e10ca9049765afaafb85a5f1cfc7c885af58045957522f661eec751e0afc36e`; the extension is
  stable5-only and cannot automatically authorize train20. Five-way one-parent array 3878 uses
  the same checkpoint and 100-iteration budget. It completes at errors
  `0.188/0.218/0.090/0.245/0.147`, with `0/5 <=0.05`; the current E4 response is therefore
  insufficient even without cross-parent conflict. The diagnostic can never authorize parent-CV.
- A scalar e3nn `l=0/1/2` tensor-product model is implemented and passes invariance, covariance,
  finite-Hessian, symmetry, parameter-gradient, and builder tests. The unscaled network has weak
  function sensitivity; scale-1000 E6 reaches median `2.630` after five steps with 10.13 GiB peak
  allocation. Half-width E7 completes 10 steps at median/P90/max `2.484/2.926/3.163`, 3.94 GiB,
  and 336 s. Formal job 3865 was stopped at step 25 as preregistered: median/max
  `2.001/2.353` after 416 s did not offset the 3--4x cost. No tensor arm passed stable5 or remains
  promoted. Five one-parent LBFGS diagnostics finish at relative-Frobenius errors
  `0.751--0.950`; this is worse than E4's `0.382--0.627` range and closes the current E7 branch.
- A new reference-local equivariant quadratic scalar predicts typed atom-pair curvature blocks,
  projects rigid modes, and derives force/HVP from `0.5 dR^T DeltaH_theta(R0) dR`. It is a local
  curvature-capacity model, not a global density-relaxed functional and not a force head. The
  rotation/translation/permutation and builder/capacity suite passes `11/11`. Stable5 smoke job
  3883 lowers median/P90/max `2.727/3.162/3.410 -> 1.499/1.851/1.968` in 20 full-batch
  steps and 23 s, with exact E/F anchoring and zero asymmetry. Formal 5000-step job 3884 is active
  on node01. Five one-parent LBFGS ceilings from step 500 finish at `0.730--1.295`, so the current
  pair-block response basis is not a capacity solution despite its speed. Train20, validation,
  train100, and Test100 remain closed.
- A separate mature higher-body scalar path now wraps isolated `mace-torch==0.3.16` from
  `/scratch/xzh/vendor/mace_torch_0_3_16`. The project environment is unchanged. MACE contributes
  only scalar energy; force and Hessian are outer autograd derivatives, and four focused tests
  cover finite/symmetric Hessians, Hessian-loss parameter gradients, rigid covariance, atom
  permutation, and scalar output scaling. Unscaled job 3891 and high-learning-rate job 3892 barely move stable5
  (`2.72677 -> 2.72610` median after 20 steps). A recorded `output_scale=1000` changes the
  five-step result to `2.71013/3.15226/3.39881`, with exact E/F anchoring, `6.0e-16` maximum
  asymmetry, 2.24 GiB peak GPU allocation, and 278 s wall time. Scaled formal stable5 job 3894 is
  active for 1000 steps on node01. This is still reference-local capacity work: no train20,
  validation, train100, or Test100 data may be read unless all five parents reach 5% and a new
  protocol is frozen.
- The one registered MACE capacity expansion, h16/correlation-3, is materially better than h8 at
  equal five-step budget: median/P90/max `2.6820/3.1118/3.3655` versus
  `2.7101/3.1523/3.3988`, with similar wall time and 3.18 GiB peak allocation. Formal h16 job
  3896 and matched h8 job 3894 are active for 1000 stable5-only steps. Promotion remains
  fail-closed on all five parents reaching 5%; early slope alone authorizes nothing.
- E4's extended full-stable5 LBFGS ceiling is final at median/P90/max
  `0.3347/0.3951/0.4133` after 100 iterations and 5541 s; every molecule remains above 23%.
  E8 is final at `0.8869/1.1578/1.2591` after 5000 steps. These are retained negative controls:
  exact anchored E/F and symmetry are not enough when the scalar response basis cannot represent
  the PBE curvature. Neither branch may proceed to train20.
- MACE h16 reaches `2.4150/2.8599/3.1162` at formal step 50 and remains ahead of h8, which is
  `2.5446/3.0592/3.2910` at step 125. Dependency job 3897 will merge the two completed runs into
  JSON/CSV/per-parent tables and a learning-curve plot. The merger is tested to reject any
  Test100-accessed input and to remain fail-closed when no model passes all stable5 gates; it never
  submits the next stage.
- The immutable h16 Adam step-75 checkpoint is frozen at
  `stage5_mace_h16_scaled_v6d/one_parent_lbfgs_v1/adam_best_snapshot.ckpt`, SHA256
  `33f5842f150d3a822fe31177f59fb7262cec781bbf217fbe2a416d79f0893229`. Stable5-only
  array 3898 runs 50 LBFGS iterations independently for each of the same five parents, four GPUs
  at a time. This tests whether shared fitting conflict or single-parent response span dominates;
  it is permanently diagnostic and cannot authorize train20. The LBFGS protocol suite passes
  `7/7`, including MACE scope and frozen-data checks.
- The first four one-parent tasks reach relative Frobenius `1.337/1.442/1.379/1.945` at LBFGS
  iteration 10, down from `2.310/2.344/2.379/3.035`; this proves movement, not 5% capacity. Dependency
  job 3903 will create a fail-closed JSON/CSV/plot summary after all array tasks complete. It
  verifies source hash, stable5 scope and permanent parent-CV prohibition.
- At iteration 20 the same four tasks are `1.112/1.260/0.993/1.221`, still far from 5%.
  Conditional job 3904 is preregistered after merger 3903: it runs a three-step h32, `l<=3`,
  three-interaction MACE smoke only if the h16 one-parent ceiling fails, otherwise it exits before
  model construction. It remains stable5-only and cannot submit formal fitting or train20.
- The completed h16 one-parent ceiling is median/P90/max `0.9184/1.0348/1.0574`, with
  `0/5 <=0.05`; individual values span `0.8072--1.0574`. This is a single-parent response-span
  failure, not merely shared fitting conflict. The read-only decision keeps parent-CV false and
  Test100 unread; aggregate JSON/CSV/plot hashes are recorded in the capacity report.
- Conditional h32 smoke is finite and symmetric but not yet better at matched budget: step-3
  median is `2.7116`, versus h16 `2.7025`. Its 323,904 trainable parameters consume 23.74 GiB
  peak GPU memory and 10:45 wall time for three steps. It remains a capacity probe.
- Stable5-only h32 one-parent array 3906 is active from the immutable step-3 checkpoint
  `1ac92a219ed416c12653bddfdef205102c879b308e60f91465e0212e0cc2b848`; dependency 3911 will
  produce a hash-bound summary. The new manifest and wrappers are
  `configs/audit/qm9_complete_total_mace_h32_one_parent_lbfgs_v1.json`,
  `scripts/slurm_qm9_complete_total_mace_h32_one_parent_lbfgs_v1.sbatch`, and
  `scripts/slurm_qm9_complete_total_mace_h32_one_parent_analysis_v1.sbatch`. Static checks,
  actual manifest binding, and focused tests pass `9/9`. This branch cannot read train20,
  validation, train100, or Test100.
- The h32 array and merger are now final: per-parent errors are
  `0.9436/1.0507/0.7553/0.9574/0.8034`, aggregate median/P90/max
  `0.9436/1.0134/1.0507`, and `0/5` pass 5%. It is slightly worse than h16 while using up to
  `23.99 GiB` per task. Do not launch h32 formal stable5 fitting.
- The MACE-compatible matrix-free Jacobian-range path is now implemented and tested. It accepts
  current stage-tagged stable5 protocols, an explicit parent ID, isolated MACE imports, and an
  explicit nonlinear E/F-anchor policy while retaining checkpoint parent-list checks. The h16
  step-75 audit on `0028399` uses only frozen stable5 artifacts.
- Eight CGLS iterations reduce the local linearized Hessian residual from `2.3104` to `1.3000`,
  but unanchored nonlinear steps catastrophically increase E/F errors. The superseding anchored
  30-iteration run reaches a linear residual of `0.9949`, nearly identical to the independent
  h16 LBFGS final `1.0009`; FD Jacobian stability is `5.3e-9` with cosine 1.0. Actual nonlinear
  Hessian probes bottom out at `1.6502` for alpha 0.3 and regress to `2.9529` at alpha 1. This
  supports a response-space/conditioning diagnosis and rejects more h16 optimizer tuning as the
  next action. Full hashes and resource numbers are recorded in the capacity report.
- The 30-iteration normal-gradient has not reached its registered tolerance, so job 3914 extends
  the same h16/`0028399` linear response audit to 100 CGLS iterations. This is not training or a
  new hyperparameter candidate; it is the final check separating ill conditioning from a true
  local response-rank floor before replacing the architecture.
- The 100-iteration audit is now final: its linear residual reaches `0.7975`, but the finite
  9.26%-norm parameter move is strongly nonlinear and alpha 1 worsens the actual anchored Hessian
  error to `13.547`. Stable finite-difference Jacobian checks (`5.5e-9`, cosine 1.0) rule out the
  matrix-free implementation as the cause. More percent-level h16 tuning is rejected.
- A replacement stable5-only arm freezes the hash-bound h16 step-75 MACE backbone and trains an
  element-resolved linear scalar readout over per-layer `l=0` invariants plus 512 fixed tanh random
  features. It has 2,720 trainable parameters, one scalar E/F/H owner, and no derivative head.
  Nine second-order/invariance/parameter-linearity tests pass. Its five-step smoke is finite and improves median
  relative Frobenius `2.7268 -> 2.6782`, with unchanged anchored E/F, `3.7e-17` asymmetry,
  `1.41 GiB` peak GPU allocation, and `161.8 s` wall time.
- Node01 jobs 3916 and 3917 are active for the unchanged 1000-step readout formal fit and an
  independent all-stable5 100-iteration matrix-free linear-subspace ceiling. Train20, validation,
  train100, and Test100 remain unopened; neither job auto-promotes.
- A measured raw-feature RMS of only `5.74e-4` explains why the unscaled random tanh kernel was
  nearly linear. The new v6g capacity arm uses rotationally invariant `l>0` channel Gram features,
  fixed tanh input scale 1000, and width 2048. Its three-step smoke improves median/P90/max to
  `2.4929/2.7632/2.8631` with unchanged anchored E/F, scalar symmetry, `0.98 GiB` peak GPU, and
  121.3 s wall time. Job 3919 is the all-stable5 exact-linear CGLS ceiling; formal optimization is
  deliberately withheld until that ceiling is known.
- Matched h8/h16 formal runs are final at median/P90/max
  `1.401/1.649/1.726` and `1.150/1.471/1.526`; neither is close to the 5% gate. The v6g exact
  all-stable5 CGLS ceiling is also final at `0.793/0.880/0.897`, while its one-parent ceiling is
  `0.2858`. These are response-span failures, not early-checkpoint or broken-Jacobian artifacts.
- Raising the frozen power-readout width from 2048 to 8192 does not improve the registered
  one-parent ceiling: 100 CGLS iterations end at `0.3077`. Do not launch an all-stable5 width-8192
  fit or increase random-feature width again from this result.
- The next active check removes the zero-force Taylor anchor and jointly solves one scalar
  `DeltaE/DeltaF/DeltaH` jet. Force is normalized as a mean component loss by `sqrt(3N)`; job 3925
  used an incorrect summed-force block and was canceled after six iterations. Superseding node01
  job 3926 is the only valid joint-jet run. It uses only stable parent `0028399`, 200 exact-linear
  preconditioned CGLS iterations, and the immutable zero-readout width-8192 checkpoint. Do not open
  all stable5, train20, validation, train100, or Test100 from it unless the one-parent Hessian is
  at most 5% and its E/F checks pass.
- Joint-jet job 3926 finishes at energy error `8.2e-7 Ha`, force MAE
  `1.342e-3 Ha/Bohr`, and Hessian relative Frobenius `0.4202` on `0028399`. The alpha-1 scalar
  evaluation matches the exact-linear solution and remains symmetric, but the normal gradient is
  not converged. Job 3928 continues from the hash-bound saved parameter step for 800 iterations;
  do not call `0.4202` a feature-span floor until this restart finishes.
- The h64 scalar force-secant 500-step pilot is final. It improves force MAE from `0.1161` to
  `0.02166 Ha/Bohr`, but complete Hessian relative Frobenius barely changes
  (`2.7268 -> 2.7139`), so ordinary summed-loss Adam is rejected. A frozen eight-direction
  parameter-gradient audit finds median E/F/HVP gradient norms `42.18/3.015/16.09` and negative
  force-HVP cosine on all directions (median `-0.236`). This is direct evidence of scale imbalance
  plus task conflict, not a broken secant graph.
- Fixed-scale PCGrad is implemented with fail-closed v6n checkpoint and audit hashes. The v6p
  fresh-Adam smoke removes gradient conflicts but oscillates energy; v6q collinear SGD with fixed
  `1e-3` steps overshoots and selects step 0. Neither formal run is allowed. The superseding v6r
  backtracking-SGD smoke accepts `5/5` common-descent updates, lowers energy error to
  `7.03e-5 Ha`, preserves scalar symmetry, and leaves force/Hessian at `0.02164/2.71385` after
  only five steps. Its formal job 3939 is stopped and archived at step 150 because strict energy
  monotonicity collapses the accepted step to `1e-8`; Hessian worsens to `2.71518`.
- The v6r step-100 correction has only `15.1%` of the target correction norm and cosine `0.104`,
  proving both amplitude and direction defects. The successor v6s uses explicit physical E/F
  budgets instead of forcing a nearly exact energy to decrease forever. Its formal run keeps
  `1e-4` steps and improves force, but full-Hessian error worsens monotonically to `2.72298` by
  step 100; job 3942 is stopped and archived. This isolates cross-direction interference in the
  one-direction stochastic updates.
- The width-8192 joint linear readout reaches energy/force `2.14e-5/4.69e-4` but only Hessian
  relative Frobenius `0.2218` after 1000 cumulative CGLS iterations. Alpha-1 agrees exactly, while
  the normal gradient remains unconverged; this is a practical compute-budget failure, not a
  mathematical rank proof.
- The six-direction force-secant run is closed. Its two-step smoke was finite at 27.56 GiB and
  slightly improved Hessian `2.713856 -> 2.713716`, but job 3945 was stopped at step 100 after the
  full Hessian moved through `2.71435/2.71494/2.71436` at steps `25/50/100`. Exposing 600 sampled
  directions did not remove matrix-level interference.
- v6u now accumulates the complete 45-direction basis without retaining all higher-order graphs.
  The frozen audit uses only 3.89 GiB and measures E/F/HVP gradient norms
  `42.1838/3.0146/4.3633`, with force-HVP cosine `-0.4547`. Its one-step smoke accepts `1e-4` and
  lowers energy/force/Hessian to `0.001803 Ha/0.021648 Ha Bohr-1/2.713781`, but the Hessian gain is
  only `7.46e-5`. Job 3949 was stopped at the registered step-5 decision: Hessian relative
  Frobenius is `2.7136175`, only `2.38e-4` below step 0, while energy/force are
  `0.0009402 Ha/0.0216112 Ha Bohr-1`. The MACE full-basis branch is closed because the observed
  slope cannot reach the 5% capacity gate within a credible budget. No stable5 or later tier is
  open from this branch.
- The replacement relaxed scalar-curvature eligibility audit is complete. Its frozen definition is
  `q(v)=v^T H v` from the same complete-total relaxed scalar energy and scalar-derived force; q is
  not a vector HVP. Job 3950 finds 74/100 parent density/branch/KKT passes, 214/400 eligible q
  directions, and 59 parents with at least three eligible directions, below the preregistered 80.
  The result is fail-closed: even removing all q checks leaves only 74 parent-gate passes. The
  dominant q diagnostic failure is the `h=1e-3` energy/force scalar closure (70 directions, 44
  low-frequency), followed by small-step stability (23) and implicit/relaxed agreement (13).
  Manifest SHA256 is `6160c8d132d0642b0e7bf0e0feb6161f7f0cde8c1e65dc203754ae41e0f8fab8`;
  Test100 access remains zero. A 59-parent run may only be labeled Stage 2.5 diagnostic. Formal
  train100 requires a preregistered, train800-only replacement-parent screen rather than relaxed
  thresholds on the observed 100.
- A train-only Stage-2.5 q diagnostic is now running without changing that failed decision. The
  source audit's 214 eligible directions include 18 directions on parents with fewer than three;
  the selected 59-parent sidecars therefore contain 196 directions. Corrected split job 3952
  freezes 137 train plus 59 one-per-parent held directions in manifest SHA256
  `c514c7fc3a550b0d5e19ddb0e35ad0d72908d627eb5523ff33be34625c3a5043`. Preflight 3953 finds
  42,027 global local-scalar features for 59 parents spanning 12--27 atoms. GPU array 3954 builds
  exact E/F/H feature jets. Index 31 (`0059755`, 27 atoms) hit a deterministic default-chunk OOM
  at 67.16 GiB resident plus a 31.14 GiB request. Old fit 3955 and analysis 3969 are cancelled.
  Rescue 3992 reruns index 31 with chunk 8; the only other 27-atom index 47 was proactively removed
  from the default array and rescue 3995 also uses chunk 8. New fit/analysis jobs must wait for an
  explicit 59/59 artifact/hash audit. This arm has base E/F for the 59 parents but not full
  train800 replay, so even a pass is only a signal
  to implement replay and strict full-Hessian checks. It cannot authorize validation or Test100.
- `scripts/qm9_complete_total_local_random_feature_stage2.py::_load_kernel_checkpoint` had been
  accidentally truncated by the v9-prior insertion. The original protocol-hash validation and
  four float64 tensor return are restored; two new regression tests and the full 13-test local
  scalar subset pass before array 3954 was submitted.
- The train700 replacement pool is also frozen before any q59 candidate result. Job 3972 assigns
  two unique candidates to each of 26 failed train100 parent-gate slots: 47/52 are exact-stratum
  and five use the deterministic nearest-stratum fallback. Manifest SHA256 is
  `a2f33838008323b66b33894d1e192c5150f244b095cfbf44c4555e61f19601da` and explicitly records
  `labels_authorized=false`. Do not generate their PBE Hessians/q labels unless Stage-2.5 passes
  its own frozen decision. The read-only post-fit analysis must be resubmitted after the replacement
  fit; canceled jobs 3955/3969 are not active work.
- The q59 Stage-2.5 diagnostic is now complete and rejected. Three exact-feature tasks required
  chunk-8 rescue; the final 59-artifact audit manifest is
  `061a2752ae603bae334bc0c4f6f6e90097e05a79142af34f2f9858681947ecfc`, with maximum feature
  `||H_asym||F/||H_sym||F=1.66e-13`. Fit job 4021 nearly interpolates all 137 train directions,
  but the selected `ridge=1e-4` arm has held median/P90 `3.777/16.524` and only `3/59 <=0.15`.
  Fit and read-only analysis hashes are
  `e9d9b1c5f8e52264b07d54563127ebf4025e05d050f08b214487f84e76d275ae` and
  `11bc52cebf714dc9d5b3c64f4e9f7ff817bfbe11dea5a9d2fe02ac9ffced2e20`.
  This is direction memorization from scalar Rayleigh constraints; no replay, replacement labels,
  full-Hessian promotion, validation, or Test100 is authorized.
- A post-fit provenance audit also found that the raw relaxed-q tasks use the original EGF
  epoch-9 checkpoint (`722afe50...d5f1f96`), whereas q59 base E/F arrays use the later A
  seed-314159 checkpoint (`e6516b04...f9d9bc`). The q59 assembled labels therefore are not
  derivatives of one common source scalar. Treat the result as an informative failed diagnostic,
  never as same-scalar evidence. The next train-only successor must bind one source checkpoint
  across E/F and relaxed vector HVP/secants and use the existing vector-stability mask; do not tune
  another ridge on the exposed q-held directions.
- That same-source relaxed-vector successor is now complete and rejected. Selection job 4023
  intersects the frozen parent gate with the frozen vector-stability mask, leaving 31 parents with
  at least three stable vector directions. It selects 20 parents by deterministic
  natoms/composition-stratum round robin, with 44 train directions and one held direction per
  parent. The 20-parent selection manifest SHA256 is
  `8585e4e6cdf56187598d4906d1459beb9e284a1a94d37cc310655405d1221d69`; its protocol hash is
  `03e3a853...2fa0`. Held directions comprise 5 angle, 7 bond, 3 low-frequency, and 5 random
  directions. Selection uses metadata and stability only, not q59 or candidate errors.
- Target-sidecar job 4024 binds all source quantities to the original random1000 EGF epoch-9
  checkpoint SHA256 `722afe50e15455fea4d516128cdf4036f5982b76fbb5e71b5a183c650d5f1f96`.
  Source HVP is the branch-mean finite difference of the complete scalar-derived relaxed force;
  PBE HVP uses the analytic PBE Hessian. The 64-direction target manifest SHA256 is
  `8152cf5e991dad4d1266bd6561b99662e5c779d626fb3e07efafcf1485669870`.
  Maximum source plus/minus branch-vector relative spread is `0.00255`, maximum PBE branch spread
  is `1.04e-16`, and maximum unit-direction norm error is `1.11e-16`.
- Source-base array 4025 and merge 4045 recompute strict complete-total relaxed energy/force for
  all 20 selected parents from that same checkpoint. All 20 succeed. The merged manifest and CSV
  SHA256 values are `409409249d6a4d1e0279e72263e40ee225d9229feef0e1c36182d97218351726`
  and `fed5adac0214d3eab019c416fdfcc46b708f26441f8a204768a0e54487b7b3f7`.
  Density relaxation takes median/max `444/10222` cycles and median/max `122/265 s`; source energy
  absolute error and force MAE medians are `0.15098 Ha` and `0.10346 Ha/Bohr`.
- Fit job 4046 solves a `5633 x 42027` float64 exact-jet design for one scalar
  `E_total=E_source+c^T phi(R)`. It takes `72.96 s`, `14.28 GiB` maximum RSS, and `3.00 GiB` peak
  GPU memory. Fit-summary SHA256 is
  `0d03784f4c035bebe59a5f3ffc8fb9bbfedad9f3d92f524e0728f1141075e828`.
  The low-ridge arms interpolate the 44 train vectors but explode out of sample: at `ridge=1e-10`
  train median/P90 is `0.000663/0.00651`, while held median/P90 is `2315/7595`. The only arm
  satisfying the train median/P90 gates, `ridge=1e-4`, gives train `0.02394/0.11921` but held
  `4.817/43.550`, `0/20 <=0.15`, and a `0.15` improvement fraction. The held-median-selected
  `ridge=1` arm gives `2.022/24.814`, `0/20 <=0.15`, and `0.75` improvement fraction, but train
  median/P90 regresses to `0.773/1.801`. Its checkpoint SHA256 is
  `ba30349aa029836dc5999c8ba3bf7830c87574e883048f3e122f63ca64d94e98`; it is diagnostic only,
  not promoted.
- This experiment closes two alternative explanations for q59: failure is not solely caused by
  scalar-Rayleigh underdetermination or mixed source checkpoints. The 42,027-dimensional v9
  scalar kernel still lacks stable unseen-direction identification under full vector constraints.
  Do not tune more ridge values, loss caps, or feature widths on these exposed 20 held directions.
  The next model must reduce the curvature hypothesis space through explicit transferable
  chemical/angular structure, freeze its choices on the 20-parent development set, and use the
  untouched 11 vector-stable parents only once as an independent train-only confirmation.

Current same-source relaxed-vector Stage-2.5 files:

```text
configs/audit/qm9_complete_total_relaxed_vector_stage2p5_split_v1.yaml
configs/audit/qm9_complete_total_relaxed_vector_stage2p5_targets_v1.yaml
configs/audit/qm9_complete_total_relaxed_vector_stage2p5_local_scalar_v1.yaml
scripts/prepare_qm9_complete_total_relaxed_vector_stage2p5.py
scripts/qm9_complete_total_relaxed_vector_sidecars.py
scripts/qm9_complete_total_relaxed_vector_local_scalar.py
scripts/qm9_complete_total_relaxed_vector_local_scalar_analysis.py
scripts/slurm_prepare_qm9_complete_total_relaxed_vector_stage2p5_v1.sbatch
scripts/slurm_qm9_complete_total_relaxed_vector_sidecars_v1.sbatch
scripts/slurm_qm9_complete_total_relaxed_vector_source_base_v1.sbatch
scripts/slurm_qm9_complete_total_relaxed_vector_source_base_merge_v1.sbatch
scripts/slurm_qm9_complete_total_relaxed_vector_local_scalar_fit_v1.sbatch
scripts/slurm_qm9_complete_total_relaxed_vector_local_scalar_analysis_v1.sbatch
tests/test_prepare_qm9_complete_total_relaxed_vector_stage2p5.py
tests/test_qm9_complete_total_relaxed_vector_sidecars.py
tests/test_qm9_complete_total_relaxed_vector_local_scalar.py
tests/test_qm9_complete_total_relaxed_vector_local_scalar_analysis.py
```

Artifacts are under
`/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_vector_train20_v1`.
Validation and Test100 access counters remain zero, train800 replay is not included, and formal
Stage 3 remains disabled.

The active v6f files are `local_mace_invariant_readout.py`, the updated MACE scalar adapter and
full-Hessian capacity CLI, protocol
`configs/audit/qm9_complete_total_hessian_mace_frozen_invariant_readout_smoke_v6f.yaml`, three
`slurm_qm9_complete_total_mace_frozen_invariant_readout_*_v6f.sbatch` wrappers, and the expanded
MACE/Jacobian tests. Reproduce only from the hash-bound smoke checkpoint recorded above.

The current unanchored joint-jet files are
`configs/audit/qm9_complete_total_hessian_mace_power_readout_joint_jet_v6j.yaml` and
`scripts/slurm_qm9_complete_total_mace_power_readout_width8192_one_parent_joint_cgls_v6j.sbatch`.
The capacity report records their exact hashes and the canceled v6i predecessor.

The current continuation and force-secant files are
`configs/audit/qm9_complete_total_hessian_mace_power_readout_joint_jet_restart_v6m.yaml`,
`scripts/slurm_qm9_complete_total_mace_power_readout_joint_cgls_restart_v6m.sbatch`,
`scripts/qm9_complete_total_mace_force_secant_capacity.py`,
`scripts/qm9_complete_total_mace_force_secant_gradient_audit.py`,
`scripts/qm9_complete_total_mace_checkpoint_hessian_geometry.py`, and the
  v6k/v6n/v6p/v6q/v6r/v6s/v6t/v6u force-secant protocols/wrappers listed in the capacity report.

New Stage-3 preparation files:

```text
scripts/prepare_qm9_complete_total_stage3_assets.py
scripts/qm9_complete_total_validation_baseline_merge.py
scripts/prepare_qm9_complete_total_active_feature_schema.py
scripts/qm9_complete_total_geometry_mlp_external_eval.py
scripts/qm9_complete_total_spectral_operator_capacity.py
scripts/qm9_complete_total_spectral_operator_external_eval.py
scripts/qm9_complete_total_spectral_operator_validation_analysis.py
scripts/qm9_complete_total_geometry_parent_cv_analysis.py
scripts/slurm_qm9_complete_total_geometry_parent_cv.sbatch
scripts/slurm_qm9_complete_total_geometry_parent_cv_analysis.sbatch
scripts/qm9_complete_total_capacity_candidate_select.py
scripts/qm9_complete_total_compare_geometry_mlp_runs.py
scripts/qm9_complete_total_external_scaling_analysis.py
scripts/qm9_complete_total_replay_descriptor_cache.py
scripts/qm9_complete_total_replay_descriptor_cache_merge.py
scripts/qm9_complete_total_replay_transient_rescue.py
scripts/qm9_complete_total_stage3_training_preflight.py
scripts/prepare_qm9_complete_total_train800_feature_inventory.py
scripts/bind_qm9_complete_total_feature_inventory.py
scripts/expand_qm9_complete_total_checkpoint_to_feature_inventory.py
scripts/expand_qm9_complete_total_hidden_width.py
scripts/expand_qm9_complete_total_deep_residual.py
scripts/launch_qm9_complete_total_atom_extensive_capacity.sh
scripts/launch_qm9_complete_total_atom_extensive_floored_capacity.sh
scripts/launch_qm9_complete_total_atom_extensive_unit_capacity.sh
scripts/launch_qm9_complete_total_floored_capacity.sh
scripts/launch_qm9_complete_total_robust_stage2_assets.sh
scripts/launch_qm9_complete_total_train800_union_assets.sh
scripts/launch_qm9_complete_total_stable5_union_assets.sh
scripts/launch_qm9_complete_total_robust_replay_cache.sh
scripts/slurm_qm9_complete_total_bind_feature_inventory.sbatch
scripts/slurm_qm9_complete_total_expand_checkpoint.sbatch
scripts/slurm_qm9_complete_total_stage3_replay_train_submit.sbatch
scripts/launch_qm9_complete_total_stage3_replay_train.sh
scripts/launch_qm9_complete_total_stage3_validation_baselines.sh
scripts/slurm_qm9_complete_total_stage3_validation_baseline_array.sbatch
scripts/slurm_qm9_complete_total_stage3_validation_baseline_merge.sbatch
scripts/slurm_qm9_complete_total_stage3_active_feature_schema.sbatch
scripts/slurm_qm9_complete_total_stage3_external_diagnostic.sbatch
scripts/slurm_qm9_complete_total_stage3_external_eval.sbatch
scripts/slurm_qm9_complete_total_stage3_external_vibrational.sbatch
scripts/slurm_qm9_complete_total_spectral_operator_capacity.sbatch
scripts/slurm_qm9_complete_total_spectral_operator_external_eval.sbatch
scripts/slurm_qm9_complete_total_spectral_operator_validation_analysis.sbatch
scripts/slurm_qm9_complete_total_stage3_replay_descriptor_cache.sbatch
scripts/slurm_qm9_complete_total_stage3_replay_descriptor_cache_merge.sbatch
scripts/slurm_qm9_complete_total_stage3_replay_merge.sbatch
scripts/slurm_qm9_complete_total_stage3_replay_rescue.sbatch
tests/test_prepare_qm9_complete_total_stage3_assets.py
tests/test_qm9_complete_total_validation_baseline_merge.py
tests/ml/test_spectral_hessian_residual.py
tests/test_qm9_complete_total_spectral_operator_capacity.py
tests/test_qm9_complete_total_spectral_operator_external_eval.py
tests/test_qm9_complete_total_spectral_operator_validation_analysis.py
configs/audit/qm9_complete_total_hessian_direction_generalization_v4_block_operator.yaml
configs/audit/qm9_complete_total_hessian_direction_generalization_v5_symmetric_completion.yaml
configs/audit/qm9_complete_total_hessian_unseen_parent_shared_operator_v1.yaml
configs/audit/qm9_complete_total_hessian_geometry_parent_cv_v1.yaml
tests/test_qm9_complete_total_geometry_parent_cv_analysis.py
mldft/ml/models/components/local_message_passing_residual.py
scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py
scripts/slurm_qm9_complete_total_local_scalar_capacity.sbatch
scripts/qm9_complete_total_local_scalar_capacity_analysis.py
scripts/slurm_qm9_complete_total_local_scalar_capacity_analysis.sbatch
scripts/qm9_complete_total_local_scalar_rigid_mode_audit.py
configs/audit/qm9_complete_total_hessian_local_scalar_capacity_v1.yaml
tests/ml/test_local_message_passing_residual.py
tests/test_qm9_complete_total_local_scalar_full_hessian_capacity.py
tests/test_qm9_complete_total_local_scalar_capacity_analysis.py
```

Remote unit checks pass `4/4` for Stage-3 asset preparation and `1/1` for validation merge failure
auditing; all three new shell entry points pass `bash -n`.

New Stage-2 files:

```text
configs/audit/qm9_complete_total_hessian_direction_generalization_v1.yaml
configs/audit/qm9_complete_total_hessian_direction_generalization_v2_dense.yaml
configs/audit/qm9_complete_total_hessian_direction_generalization_v3_near_full.yaml
configs/audit/qm9_complete_total_hessian_direction_confirmation_v1.yaml
scripts/prepare_qm9_complete_total_direction_generalization.py
scripts/qm9_complete_total_direction_confirmation.py
scripts/qm9_complete_total_stage2_select_baselines.py
scripts/slurm_qm9_complete_total_stage2_baseline_array.sbatch
scripts/slurm_qm9_complete_total_stage2_direction_mlp.sbatch
scripts/qm9_complete_total_stage2_direction_analysis.py
tests/test_qm9_complete_total_stage2_direction_analysis.py
tests/test_qm9_complete_total_direction_confirmation.py
tests/test_qm9_complete_total_geometry_mlp_capacity.py
tests/test_expand_qm9_complete_total_hidden_width.py
tests/test_expand_qm9_complete_total_deep_residual.py
tests/test_qm9_complete_total_compare_geometry_mlp_runs.py
```

## Known Limits

- P1-410 is an early pilot only.
- Historical density-relaxed Hessian metrics use finite difference of the incomplete learned-energy
  force. New `qm9_total_ofdft_hessian_audit.py` results use strict finite difference of the complete
  scalar-derived total force; they are physical total-OFDFT finite-difference Hessians, not analytic
  Hessians.
- Fixed-density autograd/HVP is finite on random1000 Test100 for all four compared models. It is a
  proxy: fixed and relaxed evaluation select the same best model on 86/100 molecules, but the
  matrices are not equivalent.
- HVP100 direct supervision can lower strict complete-total validation HVP error, but the current
  best setting regresses validation force and energy. It is diagnostic, not a promoted checkpoint,
  and Test100 remains unread for this experiment.
- Stage-2 v5 exact HVP completion is a labeled-parent matrix-completion upper bound. It is a
  conservative scalar at each supplied reference geometry, but it is not an unseen-parent OFDFT
  functional. Its shared spectral/block component fails all seven independent validation parents.
- The train20 five-fold parent-heldout audit finds useful shared signal but no promotable model.
  Unbounded parent conditioning has catastrophic tail risk; final-Hessian Frobenius caps suppress
  both the tail and most useful correction. Do not tune another cap/ridge on these same folds.
- The global geometry-scalar G0 has sufficient fit capacity but fails every held-parent Hessian
  coverage gate and the energy/vibration gates. Its preregistered replay arm G1 must not be run;
  replay cannot supply the missing cross-parent response basis.
- The local-scalar L0/L1 experiment is capacity-only and reference-geometry anchored. Even if it
  passes stable5, it is not evidence of unseen-parent transfer or a global displaced-geometry
  functional; the next parent-CV protocol remains mandatory.
- Nonlinear local random-feature v9 passes stable5 capacity and vibration gates but fails the
  frozen train20 unseen-direction gate. Its broad ridge diagnostic improves full-Hessian median
  from `1.141` to `0.365` but leaves `0/20 <=0.15` and breaks the train HVP gate. Width reduction
  fails, and maximal direction coverage still has P90 `2.50`. The all-label full20 fit reaches
  median `0.0668`, proving capacity but not generalization. Do not promote any diagnostic arm or
  run vibration, unseen-parent, train100, or Test100 evaluation from it.
- q59 cannot be used as same-scalar evidence because its base E/F and relaxed-q source checkpoints
  differ. The corrected 20-parent vector-HVP experiment does use one source scalar throughout,
  but still gives held-direction relative-L2 median `4.817` at the only train-gate-passing ridge
  and `2.022` at the held-median-selected ridge, with `0/20 <=0.15` for every arm. Full vector
  supervision therefore removes neither the interpolation/generalization gap nor the severe tail.
  Do not interpret this as a need for more directions in the same 42,027-dimensional kernel.
- The label-independent CountSketch subspace does not fix v9 direction transfer: its best arm has
  full median `0.505` and no parent below 15%. Do not tune more random subspace dimensions on the
  exposed held directions; use structured chemical/angular parameter sharing and require a newly
  frozen direction confirmation after selection.
- The chemistry-aware typed-DCT coefficient field also fails: best full median/P90 is
  `0.479/1.158` and no arm reaches 15%. Do not create a direction-confirmation split for these
  arms. The stable5 v9 coefficient prior may only be assessed with its five fitted parents marked;
  transfer claims must use the remaining 15 train20 parents.
- The stable5 v9 coefficient vector fails that unseen15 audit by two to four orders of magnitude.
  It must not initialize or regularize another Stage-2 fit.
- The structured density Hessian head proves that exact equivariant block assembly and rigid-mode
  projection are trainable, but its six-value density summary improves held median by only 0.83%.
  It is a direct matrix predictor, not a conservative total-energy Hessian or a density-response
  Schur complement.
- The frozen final-state Graphformer Hessian attention head improves its matched fold-0 held
  median by only 1.64%, wins only 1/4 parents, and worsens the held tail. The frozen final-state
  force attention head is 2.29 times worse than the existing energy-derived force and wins 0/32
  held parents against it. Neither checkpoint is eligible for Hessian initialization, backbone
  unfreezing, validation, or Test100.
- The bounded `l<=1` equivariant model is finite and trainable but fails the completed full-batch
  and one-parent LBFGS ceilings by a wide margin. E2/E4 Adam runs and the `l=2` tensor smoke remain
  capacity diagnostics only; no train20 or generalization claim is authorized until every stable5
  parent reaches 5%.
- Mature MACE h16 and h32 both fail the registered one-parent ceiling (`0/5 <=0.05`). h32 is
  slightly worse in aggregate and reaches `23.99 GiB` peak GPU allocation, so its formal stable5
  fit is closed. The active replacement is the frozen-backbone invariant linear-readout ceiling;
  it must still pass every stable5 5% gate before train20 can open.
- The h16 Jacobian-range audit shows that local linear curvature directions are numerically stable
  but strongly nonlinear and coupled to E/F when the Taylor anchor is removed. Do not interpret
  the anchored curvature model as a global complete-total functional; a promotable replacement
  must jointly fit the unanchored scalar E/F/H jet with train800 replay after first proving its
  stable5 curvature capacity.
- Mass-weighted Hessian/frequency processing accepts complete-total Hessian artifacts and reports
  frequencies, imaginary modes and mode overlap. The current complete-total result has only one
  full-matrix anchor, so it is a pilot diagnostic rather than a production frequency benchmark.
- Full-scale training code is prepared, but full-scale data I/O and full-QM9 real 8-GPU throughput are not yet validated.
- Random1000 8xA100 training validates the DDP EG/EGF path on a small remote dataset, but it is not a substitute for full-QM9 training.
- Random1000 has historical fixed/density-relaxed incomplete proxies on all 100 test molecules,
  plus five strict complete-total directional checks and one two-step full-matrix check. Do not mix
  these definitions in aggregate tables.
- Random1000 density-relaxed finite-difference Hessians have much larger antisymmetric components
  than fixed-density Hessians. Displacement, tolerance, precision, loop-integral, and relaxed
  scalar-energy audits are complete and identify the incomplete total-force definition as the
  dominant smooth-case cause; hard molecule `0040728` also has solver-branch sensitivity.
- The random1000 throughput JSON from job `479` has a stale `estimate_num_samples=4000`; use the measured wall time/samples-per-sec fields and the patched launcher for future estimates.
- GPU4PySCF force-label appending code exists; local validation has passed one-label GPU append,
  one-label CPU/GPU force comparison, append-only single-label generation, a two-worker/two-GPU
  append-only smoke, and an 8-worker/8-GPU scratch smoke on 8 labels. The 8-GPU smoke produced
  7 GPU4PySCF labels plus 1 explicitly marked `chk_derivatives` fallback. Full production-shard
  validation over many labels is still pending.
- Do not claim final P1/P2 Hessian improvement from current results.

## Recommended Next Steps

1. Freeze the 11 vector-stable parents not used by the 20-parent same-source experiment in a
   hash-bound confirmation manifest before examining any new candidate metric. Do not fit or tune
   on them. Keep the current 20 parents as the architecture-development set and use the untouched
   11 exactly once after the architecture, loss, ridge, and decision rule are frozen.
2. Keep v9 stable5, train20, q59, same-source vector20, ridge, width, CountSketch, typed-DCT, and
   v9-prior results frozen. The next architecture should be a substantially lower-dimensional,
   physically structured scalar curvature kernel with explicit shared chemical, bond, angle, and
   local-coupling parameters, not another width/ridge sweep of the 42,027-dimensional v9 basis.
   It must first pass stable5 capacity, then recover useful held-direction accuracy on the
   20-parent development set, and finally pass the untouched-11 confirmation. Keep independent
   validation and Test100 unread.
3. Keep the existing force-weight-1 and secant-weight-0.01 checkpoints as controls. For direct HVP,
   fix target semantics and directional coverage before increasing parent count; current HVP100
   and Stage-2.5 vector20 runs have no promotable candidate and must not access Test100.
4. Add a batched differentiable complete-total-energy assembly to training so force supervision is
   `-dE_total/dR`, not only `-dE_model/dR`. Keep one scalar owner and no force head.
5. Improve the tangent response solver. PCG/MINRES currently fail explicit `1e-8` residual gates
   on real 700--1200-dimensional systems and fall back to dense solves. Reuse Ritz spaces and add a
   Hartree/local-block preconditioner.
6. Add multiple directions per representative molecule before a complete-total Test100 allocation.
   Do not tune another candidate on already exposed Test100 outcomes; freeze it from validation.
7. Keep fixed float64 autograd/HVP as Tier 1, strict complete-total directional/full checks as Tier
   2, and a single frozen Test100 confirmation as Tier 3. Preserve parent grouping and split SHA.
8. Bound prepared-PySCF cache lifetime: the 16-worker Test100 stage used 28--31 GiB host memory per
   worker. Replace static cost-balanced shards with a shared task queue if future cycle counts cause
   greater than 20% worker imbalance.
9. Use the new model-parallel fixed-density launcher path; the executed serial four-model fixed
   audit took 31:19 and left most GPUs idle.
10. Keep `0000323`, `0000052`, `0040728`, `0000777`, `0072602`, `0048240`, `0129309`,
   `0024788`, `0002686`, `0053219`, `0003027`, `0004063`, `0047663`, and `0021889` as hard-case
   regressions. `0040728` is mandatory for branch checks and `0003027` for response checks.
11. For full-QM9 work, measure storage and label throughput first, run a 50--200-step 8-GPU
   calibration, and keep GPU4PySCF `records.jsonl` fallback accounting. Do not infer full-scale
   duration from random1000 without this calibration.
12. Treat reference-geometry anchoring as a local Hessian/frequency model. Before unseen-parent
   promotion, define how `R0` is supplied without labels, test displaced conservative forces
   around that fixed `R0`, and do not describe the anchored correction as a global OFDFT
   functional.
13. Keep exact symmetric completion and the parent-specific oracle only as upper bounds. The
   completed cross-parent CV proves that hard clipping is not a physical or accurate fix. A new
   representation must pass median `<=0.10`, at least 80% parents `<=0.15`, P90 `<=0.20`, and the
   frequency/imaginary-mode checks before the seven validation parents may be read again.
14. For the Graphformer-preserving derivative-head line, freeze both final-state attention heads
   as failed controls. The next small test should expose intermediate G3D edge messages or
   explicit vector channels and first perform fit-only distillation against the existing
   energy-derived force. Only a readout that can recover information already present in the
   backbone should proceed to clean PBE force labels.
15. A density-response successor must expose the pre-summation density branch or
   coefficient-level response tokens and preserve one scalar-energy owner. Do not treat more
   attention width or more steps on the final invariant atom state as a response model.

## Update Rule

When a major change happens, update this file in the same turn as the code/report change. At minimum update:

- `Last updated`;
- changed files/scripts;
- new artifacts and output directories;
- new checkpoint paths;
- metrics that affect conclusions;
- current limitations and next steps.

Do not leave major results only in chat history or `_runtime` logs.
