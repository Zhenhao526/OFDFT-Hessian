from __future__ import annotations

import json
from pathlib import Path

from .coexistence import summarize_coexistence
from .observables import observable_checklist
from .parser import parse_directory, summarize
from .trajectory import summarize_trajectories


def write_analysis_report(run_dir: Path) -> Path:
    parsed = parse_directory(run_dir)
    metadata = load_metadata(run_dir)
    trajectory_summary = summarize_trajectories(run_dir)
    report = [
        "# MPN Melting Run Analysis",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Parsed Output",
        "",
        "```text",
        summarize(parsed, natoms=metadata.get("natoms")),
        "```",
        "",
    ]
    if trajectory_summary:
        report.extend(
            [
                "## Trajectory Summary",
                "",
                "```text",
                trajectory_summary,
                "```",
                "",
            ]
        )
    coexistence_summary = summarize_coexistence(run_dir)
    if coexistence_summary:
        report.extend(
            [
                "## Coexistence Summary",
                "",
                "```text",
                coexistence_summary,
                "```",
                "",
            ]
        )
    report.extend(["## Observable Checklist", "", observable_checklist(), ""])
    out = run_dir / "analysis.md"
    out.write_text("\n".join(report), encoding="utf-8")
    return out


def load_metadata(run_dir: Path) -> dict:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
