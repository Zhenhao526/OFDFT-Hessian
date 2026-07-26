# WT Al melting point handoff

Status: verified

## Result

- Central estimate (d50): 1011.531744 K
- d50 statistical sensitivity: 1004.905811 to 1018.447122 K
- Full conservative sensitivity envelope: 953.873881 to 1086.027991 K
- Nominal discard roots: d25=1012.342241 K, d50=1011.531744 K, d75=1011.310100 K

The intervals are sensitivity envelopes, not probabilistic confidence intervals.

## Method

The free-energy sign is DeltaG = G_liquid - G_solid and the enthalpy sign is
DeltaH = H_liquid - H_solid. The calculation integrates
d(DeltaG/T)/dT = -DeltaH/T^2 from the verified 900 K absolute free-energy
anchor over the verified 900, 975, 1050, and 1100 K enthalpy grid. The melting
temperature is the root DeltaG = 0.

## Verification

All d25, d50, and d75 nominal, statistical, and conservative roots are inside
the 900-1100 K grid. Every input phase trajectory and enthalpy point passed its
physical and convergence gates before inclusion.
