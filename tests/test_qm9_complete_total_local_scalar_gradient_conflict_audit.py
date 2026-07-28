from __future__ import annotations

import torch

from scripts.qm9_complete_total_local_scalar_gradient_conflict_audit import (
    task_gradient_statistics,
)


def test_task_gradient_statistics_reports_norms_and_conflicts() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    tasks = {
        "energy": parameter[0] ** 2,
        "force": parameter[1] ** 2,
        "hessian": -(parameter[0] + parameter[1]) ** 2,
    }

    result = task_gradient_statistics(tasks, [parameter])

    assert result["gradient_norms"]["energy"] == 2.0
    assert result["gradient_norms"]["force"] == 4.0
    assert result["pairwise"]["energy_vs_force"]["cosine"] == 0.0
    assert result["pairwise"]["energy_vs_hessian"]["conflict"] is True
    assert result["pairwise"]["force_vs_hessian"]["conflict"] is True
