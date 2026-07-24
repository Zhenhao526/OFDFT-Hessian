# WT Thermodynamic Convention

Date: 2026-07-24

## Definitions

- `Delta G(T) = G_liquid(T) - G_solid(T)`.
- `Delta H(T) = H_liquid(T) - H_solid(T)`.
- Below the melting point, the solid is stable and `Delta G > 0`.
- Above the melting point, the liquid is stable and `Delta G < 0`.
- The melting point is the bracketed root `Delta G(T_m) = 0`.

The integration equation is

`d(Delta G/T)/dT = -Delta H/T^2`.

`Delta H(T)` is interpolated linearly between verified temperatures and each
segment is integrated analytically.

## Pressure And Energy

The sampled enthalpy is `H = E_total + P_external V`, where `E_total` includes
the ionic kinetic energy and WT-OFDFT potential energy. The target external
pressure is zero, so the external `P V` contribution is zero. Residual internal
pressure is an equilibration gate and is not added as a thermodynamic `P V`
correction.

Internal energy units are eV/atom. Reports use meV/atom. Temperature is in K,
pressure in kbar, volume in angstrom^3/atom, and distance in angstrom.

## Central Inputs

- 900 K absolute anchor: `Delta G = +9.3391195176 meV/atom`.
- 900 K production fusion enthalpy: `Delta H = 82.6762834433 meV/atom`.
- The `90.2827824341 meV/atom` lambda-one endpoint estimate stored in the
  anchor artifact is preliminary and must not be used in Gibbs-Helmholtz
  integration.
- The central fusion enthalpy at each temperature uses the 50% discard report.

## Uncertainty

The anchor statistical uncertainty is the RSS of the four integration-leg
block standard errors: `0.3048953417 meV/atom`.

The anchor TI conservative component is the RSS of statistical, quadrature,
and temporal terms: `0.9280610294 meV/atom`. The finite-size allowance
`3.3622904301 meV/atom` is added linearly, giving
`4.2903514595 meV/atom`. No finite-size correction is applied to the central
value.

At each enthalpy temperature:

- Statistical uncertainty = block standard error.
- Conservative uncertainty = block standard error + half drift + discard
  spread.
- At 900 K these are `2.8100590366` and `6.2786919218 meV/atom`.

Root uncertainty is reported as coherent corner-perturbation sensitivity
envelopes. These envelopes are not probabilistic confidence intervals.

## Validation

All 18 machine checks passed, including:

- Sign convention and integration direction.
- Exact `G_liquid - G_solid` reconstruction.
- Anchor and production-enthalpy value matching.
- Exclusion of the preliminary endpoint enthalpy.
- Anchor RSS and linear finite-size combination.
- Enthalpy conservative linear sum.
- Zero external-pressure contribution.
- A constant-enthalpy test with the known 1000 K root.
- Rejection of 975, 1050, and 1100 K as currently unready thermodynamic
  inputs.

Status: `verified`.
