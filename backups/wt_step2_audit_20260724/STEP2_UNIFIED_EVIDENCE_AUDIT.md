# WT Local Evidence Audit

Generated: `2026-07-24T07:59:19.468811+00:00`

This audit uses only evidence stored under `/home` on node01.

## Gates

- Required MD steps: `3000`
- Minimum nearest neighbor: `>2.0 A`
- Maximum block standard error: `3.0 meV/atom`
- Maximum discard spread: `2.0 meV/atom`
- Maximum half drift: `5.0 meV/atom`

## Temperature Status

| T (K) | Classification | Phase evidence | Enthalpy evidence |
|---:|---|---|---|
| 900 | `thermodynamic_ready` | 3000 steps; solid/liquid verified | d25/d50/d75 converged |
| 975 | `phase_ready_enthalpy_recompute_required` | final round2 3000 steps; solid/liquid verified | final round2 d25/d50/d75 absent |
| 1050 | `phase_ready_enthalpy_recompute_required` | final round2 3000 steps; solid/liquid verified | final round2 d25/d50/d75 absent |
| 1100 | `phase_ready_statistically_failed` | stitched 1300+1700 steps; solid/liquid verified | d50/d75 fail half-drift/statistical gates |

## Key Statistics

- 900 K: Delta H(d50) = 82.676 meV/atom; max block SE = 2.810; discard spread = 1.156.
- 975 K: final 3000-step phase confirmation passed, but no final round2 d25/d50/d75 enthalpy report is present locally. The earlier dataset was `extension_required` (max SE 2.548, discard spread 4.581 meV/atom).
- 1050 K: final 3000-step phase confirmation passed, but no final round2 d25/d50/d75 enthalpy report is present locally. The earlier dataset was `extension_required` (max SE 4.173, discard spread 1.777 meV/atom).
- 1100 K: phases remain valid, but exact continuation reports fail statistics: d50=82.371, d75=78.722 meV/atom; d50/d75 spread=3.649; half drifts=7.298/9.087 meV/atom.

## Decision

Overall status: `incomplete_for_gibbs_helmholtz`.

Only 900 K currently has both verified phase evidence and converged enthalpy statistics. The Gibbs-Helmholtz root must not be solved until final round2 enthalpy reports for 975 and 1050 K are regenerated from their raw trajectories.

## Next Action

When BeeGFS recovers, stage the 975/1050 K round2 `running_md.log`, `MD_dump`, and final phase JSON files into node01 `/home`; then regenerate d25/d50/d75 enthalpy reports and reapply the same gates.
