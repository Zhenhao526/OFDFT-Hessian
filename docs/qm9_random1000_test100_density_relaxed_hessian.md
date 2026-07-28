# QM9 Random1000 Test100 Density-Relaxed Hessian Evaluation

Date: 2026-07-14

## Scope

Evaluate all 100 molecules in the molecule-grouped random1000 test split with:

- EG: `qm9_random1000_eg_e10_20260713_162829/checkpoints/epoch_009.ckpt`;
- EGF lambda=1.0: `qm9_random1000_egf_lam1_e10_20260713_162829/checkpoints/epoch_009.ckpt`;
- the existing 100-molecule PBE analytic Hessian reference set.

For every Cartesian finite-difference displacement, the OFDFT density is optimized before the
model force is evaluated. Force remains derived from scalar model energy:

```text
F_pred = -dE_model/dR
```

No independent force head is used.

## Definition And Limitation

This run computes a **density-relaxed derived-force finite-difference Hessian**:

1. Optimize density at the base geometry.
2. Warm-start every displaced geometry from the converged base density.
3. Optimize the displaced density to projected gradient norm below `1e-4`.
4. Evaluate force from scalar model energy autograd.
5. Build the Cartesian Hessian with central finite difference at displacement `1e-3` Bohr.

This is not a complete analytic total-OFDFT Hessian. Classical OFDFT terms participate in density
optimization, but the current PySCF integral path is not differentiable with respect to nuclear
coordinates. Their nuclear derivatives are therefore absent from the reported force/Hessian.

## Configuration

| item | value |
| --- | --- |
| initialization | `sad_default` |
| base-density warm-start | enabled |
| stage 1 | Adam, `lr=1e-3`, `max_cycle=1000`, threshold `1e-2` |
| stage 2 | Adam, `lr=3e-4`, `max_cycle=10000`, threshold `1e-4` |
| fallback policy | `fallback-always` unless stage 1 already reaches `1e-4` |
| displacement | `1e-3` Bohr |
| parallelism | 8 external shards, one per A100 |
| node | `node04`, 8 x A100-SXM4-80GB |

The 100 molecules contain 1773 atoms in total. EG and EGF together require `21,276` displaced
density optimizations, plus 200 base-density optimizations.

## Artifacts

Slurm job:

```text
job id: 486
state: COMPLETED
exit code: 0:0
start: 2026-07-14 15:05:45
end: 2026-07-14 19:17:45
elapsed: 04:12:00
```

Output directory:

```text
/scratch/xzh/models/eval/qm9_random1000_test100_density_relaxed_hessian_node04/20260714_150546
```

Main outputs:

```text
summary.json
molecule_metrics.csv
optimization_points.csv
base_optimizations.csv
density_relaxed_stage.time.txt
hessians/*.npz
logs/shard_*.log
logs/shard_*.time.txt
```

There are 200 Hessian NPZ files: one for every model and molecule.

## Convergence

| model | base strict | displacement strict | medium | fallback triggers | mean cycles | median cycles | max cycles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EG | 100/100 | 10638/10638 | 10638/10638 | 10638 | 127.94 | 114 | 2096 |
| EGF lambda=1.0 | 100/100 | 10638/10638 | 10638/10638 | 10638 | 124.47 | 113 | 266 |

All base and displaced density optimizations reached the strict projected-gradient threshold.
There were no failed molecules, non-finite forces, or missing Hessians.

## Density-Relaxed Hessian Results

Mean of the 100 molecule-level metrics against the PBE analytic Hessian:

| model | success | MAE | RMSE | relative Frobenius | symmetry max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| EG | 100/100 | 0.070755 | 0.207097 | 3.046408 | 0.444027 |
| EGF lambda=1.0 | 100/100 | 0.012074 | 0.037043 | 0.516296 | 0.126402 |

Relative to EG, EGF lambda=1.0 has:

- `5.86x` lower Hessian MAE;
- `5.59x` lower Hessian RMSE;
- `5.90x` lower relative Frobenius error;
- `3.51x` lower mean symmetry error.

EGF is better molecule-by-molecule on MAE, RMSE, and relative Frobenius for all `100/100`
molecules. It has lower symmetry error on `84/100` molecules.

## Fixed-Density Comparison

| model | mode | MAE | RMSE | relative Frobenius | symmetry max abs |
| --- | --- | ---: | ---: | ---: | ---: |
| EG | fixed density | 0.066706 | 0.200525 | 2.921985 | 0.000408 |
| EG | density relaxed | 0.070755 | 0.207097 | 3.046408 | 0.444027 |
| EGF lambda=1.0 | fixed density | 0.010798 | 0.034852 | 0.479906 | 0.000129 |
| EGF lambda=1.0 | density relaxed | 0.012074 | 0.037043 | 0.516296 | 0.126402 |

Density relaxation does not improve agreement with PBE in this run:

- EG MAE/RMSE/relative Fro increase by `6.1%` / `3.3%` / `4.3%`;
- EGF MAE/RMSE/relative Fro increase by `11.8%` / `6.3%` / `7.6%`.

The EG/EGF ordering is nevertheless unchanged and remains strong. The much larger symmetry error
after density relaxation is important: finite-difference displacement `1e-3` amplifies residual
density-optimization noise even when every point reaches projected gradient norm below `1e-4`.
The matrices are reported without post-hoc symmetrization.

## Time Cost

| quantity | value |
| --- | ---: |
| measured evaluator wall time | 15116 s = 4:11:56 |
| Slurm elapsed | 4:12:00 |
| allocated 8-GPU time | 33.59 GPU-hours |
| EG mean molecule time | 465.84 s = 7.76 min |
| EGF mean molecule time | 671.37 s = 11.19 min |
| summed EG molecule time | 12.94 h |
| summed EGF molecule time | 18.65 h |
| EGF/EG mean molecule time ratio | 1.44x |
| shard elapsed range | 3:47:33 to 4:11:53 |
| per-shard max RSS range | about 1.56 to 1.68 GB |
| Slurm batch MaxRSS | about 13.4 GB |

The earlier fixed-density finite-difference stage for the same 100 molecules and two models took
305 seconds on 8 A100s. The strict density-relaxed stage is therefore about `49.6x` slower in wall
time. Runtime sampling showed about `1.2 GB` GPU memory per worker and low, bursty GPU utilization;
the path is dominated by CPU PySCF setup and iterative density optimization rather than A100
memory or dense GPU compute.

## Hard Cases

- Largest EG MAE: `0040728` (21 atoms), MAE `0.245101`, relative Fro `13.5344`, symmetry error
  `14.2102`. EGF reduces this to MAE `0.027813` and relative Fro `1.62970`.
- Largest EGF MAE: `0000777` (7 atoms), MAE `0.032716`, relative Fro `0.543797`.
- Slowest EG molecule: `0015263` (23 atoms), `813.5 s`.
- Slowest EGF molecule: `0060531` (27 atoms), `1062.8 s`.

## Conclusion

The random1000 Test100 result supports a robust EGF lambda=1.0 advantage over EG after strict
density optimization: EGF is better on MAE, RMSE, and relative Frobenius for every test molecule.

However, density relaxation makes both models modestly farther from the PBE reference than their
fixed-density proxies, and the finite-difference Hessian symmetry error becomes much larger. The
result should therefore be described as a **strict-converged density-relaxed derived-force Hessian
improvement in EG/EGF ordering**, not as a high-accuracy analytic physical OFDFT Hessian.

Before making stronger physical claims, validate displacement/tolerance sensitivity on the hard
cases and determine whether tighter density convergence, Hessian symmetrization, or a total-energy
implicit/analytic Hessian route removes the large antisymmetric component.
