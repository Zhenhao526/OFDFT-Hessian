# Graphformer Hybrid Relaxed-HVP Clean Rebuild V1

Last updated: 2026-07-27 18:01 Asia/Singapore

## Scope

This is a new train-only branch. The unrecoverable original-A checkpoint and old stable5/train20
manifests are not dependencies and their names or hashes must not be reused.

```text
protocol: configs/audit/qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1.yaml
protocol_id: qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1
branch_id: graphformer_hybrid_relaxed_hvp_rebuild_20260724
method: 解析密度/KKT响应的hybrid relaxed-HVP
protocol SHA256: 2b322ed75a0fa6669b79194067eaad76aea0e33e776c818e4e88ab11e7c698be
validation access: forbidden
Test100 access: forbidden
```

The method is hybrid because PySCF/libcint first integral derivatives are differentiated by a
directional central difference. Density/KKT response, total scalar energy, force, HVP and
parameter gradients remain graph-connected. It is not a fully analytic libcint second-integral
implementation.

## Frozen Data

The readable frozen source CSV has SHA256
`882a0f1da1ee74575396731156cb4b6c191b456fcddca8681db48910677f90e9` and fixes 800 train
parents with four geometries each. Validation and Test100 are empty. Because old BeeGFS label
payloads are unreadable, all 3,200 labels are recomputed with:

- PBE, `6-31G(2df,p)`, minao;
- grid level 3 with NWChem pruning;
- `def2-universal-jfit`, density-fit threshold 30;
- SCF tolerance `1e-9`;
- reference plus three deterministic perturbations, seed `20260701`;
- perturbation standard deviation `0.01 Angstrom`, cap `0.05 Angstrom`;
- scalar-derived force labels.

Node02 paths:

```text
dataset: /home/shenwei01/xzh_node02_20260724/data/QM9PBEForceRandom1000Train800RebuildV1
label run: /home/shenwei01/xzh_node02_20260724/runs/graphformer_hybrid_relaxed_hvp_rebuild_v1/labelgen
raw manifest SHA256: a009906f2545d300319dee149f7c44ab245cb6cd2c36dcbdec8bbb54a6973746
```

## New Baseline

The baseline is a seeded-random Graphformer EGF model, not original-A:

```text
name: qm9_train800_egf_force1_s12330_rebuild_v1
seed: 676368232
loss weights: energy=0.1, density-gradient=0.8, force=1.0
optimizer: AdamW, lr=7e-5, betas=(0.95,0.99), weight_decay=1e-10
effective batch: 32
fixed checkpoint: global_step=12330
selection: fixed final step, no validation
```

The checkpoint registration binds the checkpoint, protocol, experiment config and train-only
dataset manifest SHA256 values.

## HVP Definition

At a center density with projected gradient below `1e-8`, solve

```text
G_y y_v = -G_R v
H_relaxed v = L_RR v + L_Ry y_v
```

Force and HVP come from the same complete scalar total energy. Fixed-density, stale-density,
detached-evaluator and independent force/Hessian-head fallbacks are forbidden.

Each training step samples one new unnormalized Rademacher vector in the complete orthonormal
`3N-6` internal basis. Full directions are used only for periodic training-set evaluation.

## Fail-Closed Gates

Before single-parent capacity training:

- analytic HVP versus strictly reoptimized force FD relative L2 `<=1e-5`;
- parameter-gradient finite-difference relative error `<5%`;
- every audited density residual `<=1e-8`;
- full39 finite, complete and symmetry-consistent;
- same-hardware speedup `>=10x`;
- estimated training step `<60 s`;
- fastest strict response solver must also have a valid implicit parameter-adjoint path.

Before stable5:

- `0028399` internal Hessian relative Frobenius `<=0.05`;
- energy and force errors each regress by no more than 5% from the new baseline.

Train20 remains locked until a separately registered stable5 gate passes.

## Execution

The active label reconstruction uses:

```bash
bash scripts/launch_qm9_train800_fresh_labels_node02.sh
```

The downstream watcher/orchestrator uses:

```bash
bash scripts/launch_qm9_graphformer_hybrid_rebuild_node02.sh
```

It creates new PBE Hessian, parent, direction, checkpoint, audit and capacity registrations under
`/home/shenwei01/xzh_node02_20260724/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1`.

GPU PBE references use the isolated runtime through:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> scripts/run_gpu4pyscf_node02.sh \
  scripts/qm9_train800_pbe_hessian_references.py ...
```

The runtime is
`/home/shenwei01/xzh_node02_20260724/envs/gpu4pyscf-cuda12-py311` with
GPU4PySCF `1.7.4`, CuPy `14.1.1`, and PySCF `2.13.1`. It does not modify the main project
environment, which remains on PySCF `2.4.0`.

## Current Status

- Kohn-Sham completed and verified `3200/3200` `.chk` in `31:28:46`.
- Label generation completed `3200/3200` in `13:56:01`.
- Force validation passed `3200/3200`, with zero failures and maximum force norm
  `0.13713889631457066 Ha/Bohr`.
- The corrected transform completed `3200/3200`; split writing and dataset-statistics generation
  also completed.
- The final-registration `NameError: name 'split' is not defined` was fixed without rerunning
  Kohn-Sham, label generation or transforms. The focused finalization tests pass.
- The registered train-only dataset contains 800 parents, 3,200 label files and 39,456
  hash-selected SCF-expanded training samples. Validation and Test100 are empty. Every numeric
  value in the raw and transformed labels is finite.
- Dataset manifest SHA256:
  `9506bf6573ecdea705555d204c389ce7508735baccc3c13533a7b23c623370ed`.
- Split SHA256:
  `5b58f73136bbda780340002cb3a356488a6c4cc6b6df1d94b054a352c4d5fb1c`.
- The first GPU4PySCF PBE Hessian attempt failed `0/20` before GPU4PySCF installation. Its failed
  manifest is preserved and forbidden; it was not reused.
- A separate GPU4PySCF runtime is now installed and validated on node02. CuPy arithmetic and a
  finite, symmetric H2/PBE analytic Hessian pass. The environment package/hash registration list
  has SHA256
  `3928950929fee10dd9bb7e885759c081e03eac3fa6cf12164a86a6d3da9a9d3b`.
- `0028399` passes the GPU4PySCF versus CPU PySCF analytic-Hessian gate. GPU/CPU relative
  Frobenius disagreement is `1.41937e-4`, GPU symmetry max abs is `4.44e-16`, and the GPU run is
  `36.8x` faster on the measured node02 runs.
- The first CPU helper invocation added PySCF CPHF scratch groups to the source `.chk`. That
  mutated copy is preserved as a failed audit artifact; only the newly added groups were removed,
  restoring the original semantic payload hash. The helper now sets `mf.chkfile=None` after SCF
  restoration and fails if the source byte hash changes. The formal GPU rerun left it unchanged.
- The formal train20 PBE reference now has exactly 20 successes and zero failures. All Hessians
  are finite, maximum symmetry error is `1.776e-15`, wall time is `1187.77 s`, and manifest
  SHA256 is
  `acc5d00a984f17f73043aafcc5954fba7722537ab55c82f803a64f99a61f30c1`.
- New direction assets are frozen. Stable5 has 228 complete internal directions and manifest
  SHA256
  `e1d456e8c5dced55461a3067af3e1eaa158422e9551e982a364ba63b4f75eb1b`;
  train20 has 906 directions and manifest SHA256
  `1ea9fe04a971b5998620d49f5e68567938ab4f505f17636fb2065085b8c73804`.
- The first baseline launch was stopped fail closed at step 431 because Lightning's automatic
  distributed-sampler reconstruction silently changed runtime batch size from 4 to 1. It has no
  registered checkpoint and is preserved under `models/train/runs/failures/`.
- `trainer.use_distributed_sampler=false` now preserves the registered loader. A separate two-step
  smoke observed `(per-GPU batch, accumulation, effective batch)=(4,8,32)`. Final checkpoint
  registration also validates this tuple from the throughput CSV.
- The fresh fixed-step baseline completed from step zero through exact step 12,330 in `2:17:10`.
  Its checkpoint SHA256 is
  `ca45fda1eedb9815e9cac26c23f95f0a60666f73c89d4a0e967803ddc4690a45`.
  Registration observed 9,864 throughput rows with runtime
  `(per-GPU batch, accumulation, effective batch)=(4,8,32)`. Selection was fixed final step
  without validation.
- The initial analytic audit failed with HVP-vs-FD relative L2 `388.69`, despite density response
  and parameter-gradient checks passing. Component audits proved two independent second-order
  graph defects:
  - transformed Hartree and nuclear-attraction paths had catastrophic second-derivative
    cancellation, while the same physical-basis scalar classical energy matched scalar FD to
    `5.23e-7`;
  - the custom natural-reparametrization matrix square root supplied a stable first derivative
    but did not preserve the input dependency needed for a correct double backward.
- `evaluate_total_ofdft_force` now assembles Hartree, nuclear attraction and nuclear repulsion
  directly in the untransformed physical basis. Graphformer `kin_plus_xc` remains the same scalar
  model. The explicit `eigh_second_order_audit` matrix-power mode preserves the second-order
  natural-reparametrization graph. It is not a silent fallback and is recorded in audit outputs.
- At strict force-FD displacement `1e-4 Bohr`, the corrected single-direction audit gives:
  - analytic relaxed-HVP versus strictly reoptimized force FD relative L2 `3.50e-7`;
  - fixed-density partial versus force FD relative L2 `3.14e-9`;
  - density-response correction versus FD relative L2 `3.60e-9`;
  - maximum projected density-gradient norm `9.05e-11`.
- The formal `0028399` audit completed full39 and failed closed only on performance:
  - analytic HVP versus strict reoptimized FD `3.339e-7`, pass;
  - parameter-gradient FD relative error `3.836e-4`, pass;
  - maximum density residual `9.943e-11`, pass;
  - full39 complete, relative Frobenius `2.576`, symmetry ratio `1.203e-5`;
  - estimated analytic training step `111.97 s`, above the `60 s` gate;
  - measured speedup over the same-hardware four-direction legacy path `0.882x`, below `10x`;
  - peak GPU allocation `67,860 MB`, max RSS `77,925 MB`, total wall `47:15`.
- Dense direct is the only strict, parameter-adjoint-capable response solver:
  residual `7.88e-15` in `20.75 s`. PCG, MINRES and deflated-PCG did not meet the registered
  convergence and response-agreement gates.
- Formal summary SHA256:
  `d38988ca3463dded5cec854ff449600055e783405d96bb586d010e56156a7a47`.
- `single_parent_capacity_training_allowed=false`,
  `curvature_corrector_evaluation_required=true`, and `stable5_train20_allowed=false`.
  No capacity, stable5 or train20 training was started.
- Final provenance lists are
  `provenance/code_snapshot_sha256.txt`,
  `provenance/analytic_hvp_audit_registered_sha256.txt`, and
  `provenance/final_asset_invariants_20260727.sha256` under the branch artifact root.
- No validation or Test100 record has been accessed.

## Reproduction Commands

The registered baseline was launched with:

```bash
CUDA_VISIBLE_DEVICES=7 \
  bash scripts/launch_qm9_graphformer_rebuild_baseline_node02.sh
```

The accepted strict HVP/FD step check was:

```bash
MLDFT_SYMMETRIC_MATRIX_POWER_MODE=eigh_second_order_audit \
python scripts/qm9_graphformer_relaxed_hvp_component_audit.py \
  --protocol configs/audit/qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1.yaml \
  --checkpoint-registration \
    /home/shenwei01/xzh_node02_20260724/models/train/runs/qm9_train800_egf_force1_s12330_rebuild_v1/checkpoint_registration.json \
  --parent-manifest \
    /home/shenwei01/xzh_node02_20260724/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/manifests/stable5_parent_manifest.json \
  --direction-manifest \
    /home/shenwei01/xzh_node02_20260724/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/directions/stable5/manifest.json \
  --output-dir \
    /home/shenwei01/xzh_node02_20260724/runs/graphformer_hybrid_relaxed_hvp_rebuild_v1/audit_0028399_eigh_fd1e4_v1 \
  --molecule 0028399 \
  --direction-index 0 \
  --force-fd-displacement 1e-4 \
  --directional-second-steps 1e-3 \
  --device cuda:0
```

The formal audit used:

```bash
python scripts/qm9_graphformer_analytic_relaxed_hvp_audit.py \
  --protocol configs/audit/qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1.yaml \
  --checkpoint-registration \
    /home/shenwei01/xzh_node02_20260724/models/train/runs/qm9_train800_egf_force1_s12330_rebuild_v1/checkpoint_registration.json \
  --parent-manifest \
    /home/shenwei01/xzh_node02_20260724/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/manifests/stable5_parent_manifest.json \
  --direction-manifest \
    /home/shenwei01/xzh_node02_20260724/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/directions/stable5/manifest.json \
  --output-dir \
    /home/shenwei01/xzh_node02_20260724/runs/graphformer_hybrid_relaxed_hvp_rebuild_v1/audit_0028399_formal_eigh_physical_classical_v1 \
  --molecule 0028399 \
  --direction-index 0 \
  --strict-fd-displacement 1e-4 \
  --symmetric-matrix-power-mode eigh_second_order_audit \
  --device cuda:0
```

## Decision

The corrected hybrid analytic path is a valid small-sample correctness oracle on `0028399`.
It is not a usable HVP training implementation: one step remains above one minute, native
second-order `eigh` retains about 68 GB on GPU, and full39 takes about 26.7 minutes. The next
implementation branch should evaluate an explicit scalar, energy-conserving graph curvature
corrector with train800 E/F replay. It must not inherit a capacity checkpoint from this failed
audit and must keep validation and Test100 closed.
