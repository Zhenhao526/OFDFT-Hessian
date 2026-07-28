# QM9 random1000 HVP branch-stability protocol v2

Frozen: 2026-07-16, after the preregistered v1 smoke and before the eight-molecule diagnostic or
any train100 stability result. Machine-readable configuration:
`configs/audit/qm9_hvp_branch_stability_v2.yaml`.

This document incorporates all unchanged data boundaries, density branches, stability gates,
aggregation rules and promotion rules from
`docs/qm9_random1000_hvp_branch_stability_protocol.md`. Test100 remains unread.

## Single protocol correction

V1 used the same `h={3e-5,1e-5,3e-6}` Bohr points for force-derived HVP stability and relaxed
scalar-energy second differences. The `0000751`, direction-0 smoke showed that this is numerically
invalid for the energy audit: the three density branches agreed to `2.3e-13 Ha`, Coulomb density
distance was `5.2e-12`, complete-force branch RMSE was `2.7e-11 Ha/Bohr`, KKT condition number was
`9.81e6`, and force-derived HVP step instability was only `4.05%`. Nevertheless approximately
`1e-9 Ha` endpoint energy noise, divided by `h^2`, produced scalar-energy curvatures from `-1.60`
to `363.7 Ha/Bohr^2`, while force-derived curvature remained near `-0.8 Ha/Bohr^2`.

V2 therefore keeps the original three steps exclusively for strict relaxed-force HVP stability.
The relaxed-energy versus relaxed-force conservative closure check uses a separate `1e-3 Bohr`
step; only steps `>=3e-4 Bohr` enter that gate. No other threshold changes. V1 remains preserved as
the audit trail and its smoke classification is not reused for model selection.
