# T1100 WT Al108 continuation and stitch audit

Audit time: 2026-07-24

## Scope

- Historical segment: global steps 0-1300.
- Recovered restart frame: global step 1300.
- Local continuation: 1700 MD steps, mapped to global steps 1300-3000.
- Solid and liquid contain 108 Al atoms each.
- Ensemble: NVT, CSVR tau = 5 fs, target temperature = 1100 K.
- MD time step = 1 fs; trajectory dump interval = 5 steps.

## Provenance and stitch rule

The original node04 trajectory was formally paused at complete frame 1350, but only
the complete step-1300 restart structures survived in the local backup. The new
continuation therefore branches from global step 1300. Historical frames 1305-1350
must not be counted.

The original 0-1300 raw `MD_dump` and `running_md.log` are not present in the node01
restore tree or the local backup. A single physical raw trajectory file cannot be
reconstructed without inventing missing records. The valid logical mapping is:

- Historical frames: global 0-1300.
- Continuation frame local 0: duplicate representation of global 1300.
- Continuation frames local 5-1695: global 1305-2995.
- Continuation thermodynamic rows local 1-1696: global 1301-2996.
- The MD integrator reached local step 1700, hence global target step 3000.

The global discard-50% interval (1500-3000) and discard-75% interval (2250-3000)
are fully contained in the continuation. Their statistics are exact. Discard-25%
cannot be reconstructed exactly because it requires unavailable historical raw rows.

## Boundary integrity

The recovered `STRU_MD_1300` and the generated continuation input `STRU` agree:

- atom count: 108 for each phase;
- atom ordering and symbols: identical;
- maximum position mismatch: below 1.0e-11 A;
- velocity mismatch: exactly 0 A/fs;
- maximum lattice mismatch: below 4.2e-13 A.

The first dumped frame is displaced from the source by only about 6e-6 A RMS,
consistent with the first integration update. No restart discontinuity was detected.

## Trajectory completeness

Both phases have:

- 1700 logged MD steps and a normal ABACUS `FINISH`;
- 340 unique complete frames at local steps 0, 5, ..., 1695;
- 340 unique thermodynamic rows at local steps 1, 6, ..., 1696;
- 108 atoms in every parsed frame;
- finite thermodynamic values and no NaN, abort, or convergence failure.

Electronic minimization:

- solid: 10.25 iterations/step mean, maximum 13;
- liquid: 10.46 iterations/step mean, maximum 14;
- no step reached `scf_nmax = 100`.

## Exact global production statistics

### Global discard 50%: steps 1500-3000

| quantity | solid | liquid |
|---|---:|---:|
| retained samples | 300 | 300 |
| temperature (K) | 1094.20 +/- 86.80 | 1094.29 +/- 87.55 |
| pressure (kbar) | -1.605 +/- 3.455 | +1.016 +/- 2.762 |
| final nearest neighbor (A) | 2.374 | 2.339 |
| minimum nearest neighbor (A) | 2.202 | 2.140 |

The solid-liquid mean temperature difference is 0.09 K. Both mean pressures are
within the established 2.5 kbar zero-pressure tolerance.

### Global discard 75%: steps 2250-3000

| quantity | solid | liquid |
|---|---:|---:|
| retained samples | 150 | 150 |
| temperature (K) | 1108.32 +/- 88.10 | 1107.66 +/- 89.85 |
| pressure (kbar) | -0.450 +/- 3.426 | +1.109 +/- 2.924 |

## Phase integrity

Solid:

- status: `solid_verified`;
- final non-affine MSD: 0.319 A2;
- late MSD slope: -5.89e-5 A2/step;
- final CSP median: 4.019 A2;
- recent-20-frame CSP median mean: 5.035 A2;
- recent ordered fraction mean: 0.162.

Liquid:

- status: `liquid_verified`;
- final non-affine MSD: 6.365 A2;
- late MSD slope: 3.186e-3 A2/step;
- diffusion MSD slope: 3.837e-3 A2/step;
- final CSP median: 15.697 A2;
- recent-20-frame CSP median mean: 15.286 A2;
- recent ordered fraction mean: 0.

No collapse or phase interchange occurred.

## Fusion-enthalpy convergence

Using sampled total energy at zero external pressure:

- global discard 50%: 0.082371 eV/atom, block SE 0.002911 eV/atom;
- global discard 75%: 0.078722 eV/atom, block SE 0.003673 eV/atom;
- d50-d75 spread: 0.003649 eV/atom.

The strict first-half/second-half drift checks are 7.30 and 9.09 meV/atom,
respectively, above the configured 5 meV/atom threshold. Ten consecutive blocks
range from 67.96 to 89.54 meV/atom and are oscillatory rather than monotonic.

Therefore the trajectories are structurally and thermodynamically valid, but the
1100 K fusion-enthalpy point remains `gate_failed` under the strict convergence
criterion. It should not yet be treated as a final converged Gibbs-Helmholtz input.

## Files on node01

- `solid/phase_analysis_resume1700.json`
- `liquid/phase_analysis_resume1700.json`
- `trajectory_integrity_and_thermo_resume1700.json`
- `analysis_manifest.json`
- `global_d50_exact.json`
- `global_d75_exact.json`

