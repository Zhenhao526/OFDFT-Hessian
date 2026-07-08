# MPN-OFDFT Al/Mg Melting Workflow

This workspace starts the local execution of the MPN-OFDFT melting plan.

Current scope:

- Generate fcc Al and hcp Mg supercells.
- Write ABACUS `INPUT`, `STRU`, `KPT`, `metadata.json`, and `run_local.sh`.
- Prepare static, liquid-preparation, and coexistence-seed jobs.
- Parse basic ABACUS output when calculations are available.

The first executable target is pure Al/Mg. Al-Mg alloy generation is intentionally left for the next phase after pure-element validation.

## GPU Remote Target

The production route uses the GPU-enabled ABACUS MPN-OFDFT executable, not the CPU-only binary. The remote development copy is expected at:

```bash
/mnt/afs/home/xiazhenhao/alloy_mpn_ofdft_dev_20260708
```

The remote GPU config is `config/abacus_mpn_gpu_remote.json`. It points to:

```bash
/mnt/afs/home/xiazhenhao/alloy_mpn_ofdft_dev_20260708/external/abacus-develop/build-gpu-gcc13/source/abacus_ml_gpu
```

Use this config for A100 runs with `of_ml_device gpu`. The MPN model and pseudopotentials are expected under `assets/mpn/net.pt` and `assets/pseudo/` on the remote machine; these binary/input assets are not committed here.

ABACUS has been built locally at:

```bash
/Users/xia/Documents/合金/external/abacus-develop/build-mpn/source/abacus_pw_para
```

For MPN-OFDFT, use the ML-enabled binary:

```bash
/Users/xia/Documents/合金/external/abacus-develop/build-mpn/source/abacus_ml_para
```

The default config uses ABACUS' bundled `tests/PP_ORB` directory and BLPS pseudopotentials for the first MPN-OFDFT smoke tests.

## Quick Start

```bash
python3 -m mpn_melting env-check
python3 -m mpn_melting init
python3 -m mpn_melting make-static --element Al --size 2 2 2 --out runs/al/static/fcc_2x2x2
python3 -m mpn_melting make-static --element Mg --size 2 2 2 --out runs/mg/static/hcp_2x2x2
python3 -m unittest discover -s tests
```

Short MD equilibration checks can be generated with explicit thermostat options:

```bash
python3 -m mpn_melting make-liquid --element Al --size 2 2 2 --temperature 1200 --steps 20 --out runs/al/liquid/T1200_20step
python3 -m mpn_melting make-liquid --element Al --size 2 2 2 --temperature 1200 --steps 20 --thermostat rescale_v --nraise 1 --seed 20260707 --out runs/al/liquid/T1200_20step_rescalev
```

Use `rescale_v` only as a preheating/control check. The short Lindemann scan is not a melting-point method; it is useful only for debugging that MD, forces, and trajectory parsing work.

## Solid-liquid coexistence workflow

The production melting workflow is now based on two-phase coexistence:

1. Pre-melt a liquid seed at a high temperature. If the seed is too aggressive or too short, restart `make-liquid` from the last `MD_dump` frame with `--source` and anneal it at a lower liquid temperature.
2. Build an elongated coexistence box with the lower half crystalline and the upper half from the pre-melted liquid frame.
3. Run short NVT equilibration if needed, then NVE at candidate temperatures.
4. Use `analysis.md` to inspect the `Coexistence Summary`: liquid growth means the temperature is above the melting point, solid growth means it is below, and a near-stationary interface brackets the melting point.

Example smoke test:

```bash
python3 -m mpn_melting make-coexist \
  --element Al --size 2 2 4 --temperature 940 --steps 5 --ensemble nve \
  --seed 20260707 \
  --liquid-source runs/al/liquid/T3000_20step_rescalev \
  --abacus-config config/abacus_mpn.json \
  --out runs/al/coexist/mpn_T0940_2x2x4_nve5_from_T3000_unwrapped
```

The 5-step example is only a smoke test. For melting estimates, extend the coexistence trajectories and bracket temperatures around the experimental range, for example 880, 920, 940, 980, and 1020 K for Al.

To run a generated job:

```bash
cd runs/al/static/fcc_2x2x2
./run_local.sh
```
