from __future__ import annotations

import math
from typing import Sequence, Tuple


EnthalpyPoint = Tuple[float, float]


def integrate_linear_enthalpy_over_t2(
    temperature_left_k: float,
    temperature_right_k: float,
    enthalpy_left_ev: float,
    enthalpy_right_ev: float,
) -> float:
    """Integrate a linearly interpolated enthalpy divided by T**2."""
    if temperature_left_k <= 0.0 or temperature_right_k <= temperature_left_k:
        raise ValueError("temperatures must be positive and increasing")
    slope = (enthalpy_right_ev - enthalpy_left_ev) / (
        temperature_right_k - temperature_left_k
    )
    intercept = enthalpy_left_ev - slope * temperature_left_k
    return intercept * (1.0 / temperature_left_k - 1.0 / temperature_right_k) + slope * math.log(
        temperature_right_k / temperature_left_k
    )


def gibbs_over_temperature(
    anchor_temperature_k: float,
    anchor_delta_g_ev: float,
    enthalpy_points: Sequence[EnthalpyPoint],
) -> list[EnthalpyPoint]:
    """Return DeltaG/T at each temperature using Gibbs-Helmholtz integration."""
    points = sorted((float(t), float(h)) for t, h in enthalpy_points)
    if len(points) < 2:
        raise ValueError("at least two enthalpy points are required")
    if any(right[0] <= left[0] for left, right in zip(points, points[1:])):
        raise ValueError("enthalpy temperatures must be unique")
    anchor_matches = [
        index
        for index, (temperature, _) in enumerate(points)
        if math.isclose(temperature, anchor_temperature_k, abs_tol=1.0e-9)
    ]
    if len(anchor_matches) != 1:
        raise ValueError("one enthalpy point must be at the anchor temperature")
    anchor_index = anchor_matches[0]
    values = [0.0] * len(points)
    values[anchor_index] = anchor_delta_g_ev / anchor_temperature_k

    for index in range(anchor_index, len(points) - 1):
        left_t, left_h = points[index]
        right_t, right_h = points[index + 1]
        values[index + 1] = values[index] - integrate_linear_enthalpy_over_t2(
            left_t, right_t, left_h, right_h
        )
    for index in range(anchor_index - 1, -1, -1):
        left_t, left_h = points[index]
        right_t, right_h = points[index + 1]
        values[index] = values[index + 1] + integrate_linear_enthalpy_over_t2(
            left_t, right_t, left_h, right_h
        )
    return [
        (temperature, values[index])
        for index, (temperature, _) in enumerate(points)
    ]


def solve_melting_temperature(
    anchor_temperature_k: float,
    anchor_delta_g_ev: float,
    enthalpy_points: Sequence[EnthalpyPoint],
    tolerance_k: float = 1.0e-6,
) -> float:
    """Solve DeltaG(Tm)=0 inside a bracketed enthalpy-temperature series."""
    points = sorted((float(t), float(h)) for t, h in enthalpy_points)
    values = gibbs_over_temperature(anchor_temperature_k, anchor_delta_g_ev, points)
    for index, ((left_t, left_y), (right_t, right_y)) in enumerate(
        zip(values, values[1:])
    ):
        if left_y == 0.0:
            return left_t
        if left_y * right_y > 0.0:
            continue
        left_h = points[index][1]
        right_h = points[index + 1][1]
        base_t = left_t
        base_y = left_y
        base_h = left_h
        lo = left_t
        hi = right_t
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fraction = (mid - base_t) / (right_t - base_t)
            mid_h = base_h + fraction * (right_h - base_h)
            mid_y = base_y - integrate_linear_enthalpy_over_t2(
                base_t, mid, base_h, mid_h
            )
            if abs(hi - lo) <= tolerance_k:
                return mid
            if base_y * mid_y <= 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)
    raise ValueError("enthalpy temperatures do not bracket a melting root")
