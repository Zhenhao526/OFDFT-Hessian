# QM9 random1000 Graphformer residual experiment v1

## Objective

Test whether the existing 18,692,586-parameter Graphformer is easier to train as a
correction to APBEK than as a direct model of `Ts + Exc`. The architecture, split,
random seed, density states and optimizer schedule are held fixed.

The residual model represents

```text
E[n, R] =
    DeltaTs_Graphformer[n, R]
  + T_APBEK[n, R]
  + Exc_PBE[n, R]
  + EH[n, R]
  + Eext[n, R]
  + Enn[R].
```

It learns only `DeltaTs = Ts - T_APBEK`. PBE XC remains a density functional
evaluated on the numerical grid; no Kohn-Sham orbitals are used during OFDFT
inference.

## Frozen comparison

The historical random1000 identity is restored without opening old damaged
storage:

- parents: QM9 IDs 1 through 1000;
- parent split: NumPy legacy permutation with seed 8;
- train/validation/test: 800/100/100 parents;
- four geometries per parent;
- train parent identity is checked against the immutable rebuilt train800 split.

The three comparison arms are:

1. zero learned residual: APBEK + PBE XC;
2. direct Graphformer: learn `Ts + Exc`;
3. residual Graphformer: learn `Ts - T_APBEK`, then add APBEK + PBE XC.

For the residual arm, the error of the learned residual is exactly the error of
the reconstructed `Ts + Exc` term, because the same deterministic APBEK and PBE
XC values are added to prediction and reference. Energy and projected
density-gradient metrics are therefore directly comparable across arms.

## Why force loss is disabled in this first comparison

The historical `ForceLoss` differentiates only the scalar produced by the
Graphformer and compares that derivative with the complete PBE nuclear force.
That is already incomplete for a direct `Ts + Exc` model and is unambiguously
wrong for a `Ts - T_APBEK` residual: it omits explicit geometry derivatives of
APBEK, PBE XC, Hartree, electron-nuclear attraction and nuclear repulsion.

This experiment therefore trains energy and density-functional derivatives
only. If the residual passes the frozen representation gate, physical force and
Hessian validation uses finite differences of the complete, density-relaxed
total energy. No fixed-density force is reported as an OFDFT force.

## Blind-test boundary

Validation100 and Test100 are rebuilt into separate dataset roots. The default
`prepare`, `train`, `eval` and `all` stages do not generate, transform, inspect
or include Test100 labels. They write only `split_train_val.pkl`.

After architecture, checkpoint selection and thresholds are frozen, a
`protocol_frozen.json` record is written under the run root. Only then may the
launcher run with `STAGE=test`; it rebuilds the separate Test100 root, invokes
the split script with `--unlock-test`, writes `split_full.pkl`, and evaluates
Test100 once. Test100 is not used for model selection.

## Run

On node02 local storage:

```bash
export FULL_QM9_RAW_DIR=/home/shenwei01/xzh_node02_20260724/data/QM9/raw
export RESIDUAL_GPU=7
STAGE=all bash scripts/launch_qm9_residual_random1000_node02.sh
```

After validation is reviewed, freeze the exact checkpoints, validation files and
gate decision:

```bash
STAGE=freeze bash scripts/launch_qm9_residual_random1000_node02.sh
```

This writes `qm9_residual_random1000_v1/protocol_frozen.json`. Only then run:

```bash
STAGE=test bash scripts/launch_qm9_residual_random1000_node02.sh
```

If a complete QM9 raw directory is unavailable, the launcher downloads the
official QM9 archive into node02 local `/home` and extracts only the frozen
validation or test parent files. It fails closed if the historical split
identity drifts, a partition is incomplete, or the selected GPU is occupied. It
never reads or writes `/scratch`.

The machine-readable protocol and gates are in
`configs/audit/qm9_residual_random1000_graphformer_v1.yaml`.
