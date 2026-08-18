# Mg XWM Melting Results (2026-08-15)

This directory preserves the compact, machine-readable result bundle for the
Mg XWM melting workflow and its KSDFT single-direction diagnostic correction.
Raw trajectories and ABACUS output directories remain outside Git.

## Main results

- Reporting status: preliminary XWM analysis with user-authorized gate waivers.
- XWM free-energy anchor at 900 K:
  `Delta G(liquid-solid) = 8.7464411021 meV/atom`.
- Primary d50 XWM melting temperature: `1001.0929 K`.
- Conditional d50 bootstrap mean and 95% interval:
  `1001.1006 K [985.4660, 1017.4964]`.
- Conservative anchor uncertainty: `7.1255 meV/atom`.
- KSDFT one-way direct free-energy root: `1001.5651 K`.
- KSDFT-corrected enthalpy-integration root: `998.1482 K`.
- Minimum one-way FEP ESS fraction: `0.29275`.

The KSDFT correction is a single-direction diagnostic from XWM ensembles. It
does not include a reverse KS ensemble and must not be described as a strict
bidirectional free-energy correction. The bootstrap intervals are conditional
statistical diagnostics and exclude the full conservative XWM anchor systematic
allowance.

## Files

- `preliminary_anchor_and_tm.json`: XWM anchor, enthalpy integration, melting
  roots, bootstrap, uncertainty budget, and provenance.
- `analytic_reference_free_energies.json`: analytic reference free energies
  used by the absolute anchor.
- `user_gate_waiver.json`: exact scope of the user-authorized gate waivers.
- `PILOT_VALIDATION_COMBINED.json`: six-point KSDFT pilot validation.
- `oneway_diagnostic_analysis.json`: 60-snapshot, three-temperature KSDFT
  one-way FEP and internal-energy correction analysis.
- `Mg_XWM_melting_report_20260815.pdf`: XWM result report.
- `Mg_WT_XWM_melting_report_20260815.pdf`: WT/XWM comparison report.
- `SHA256SUMS`: checksums for the preserved JSON and PDF payload.

The three core JSON files were copied from the audited node01 workspace. Their
SHA-256 checksums match the remote results:

```text
80e80a1bbb46e7d800a05b4b25c12b1960f637c2e5fff30f283e6b7dc33b9c77  preliminary_anchor_and_tm.json
e4c512719dd55ee4d2c8eede61f3ccb6ce9810649771ec00663f69ca6c67b8bc  oneway_diagnostic_analysis.json
c22c0da70d22581db4a290e70588176d3d5b44b840baed5c85f63854c7e0a943  user_gate_waiver.json
```
