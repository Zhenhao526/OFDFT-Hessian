import json
import os
from pathlib import Path

from scripts.verify_kedf_ti_formal_preflight import sha256, verify


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_grid(tmp_path: Path) -> Path:
    root = tmp_path / "formal"
    pair = tmp_path / "pair.dat"
    merged = tmp_path / "merged.json"
    pair.write_text("pair\n")
    merged.write_text("{}\n")
    sources = []
    for phase in ("solid", "liquid"):
        windows = []
        for index in range(9):
            label = f"lambda_{index:02d}"
            point = root / phase / label
            source_structure = tmp_path / "sources" / phase / label / "MD_dump"
            write(source_structure, f"{phase} {label}\n")
            tau = 2.0 if (phase, label) == ("solid", "lambda_02") else 5.0
            metadata = {
                "csvr_tau": tau,
                "mpi_ranks": 12,
                "pair_model_sha256": sha256(pair),
                "source_step": 295,
                "source_structure_sha256": sha256(source_structure),
                "source_velocities_discarded": False,
                "steps": 3000,
                "target_kedf": "lkt",
            }
            write(point / "metadata.json", json.dumps(metadata))
            write(
                point / "INPUT",
                "INPUT_PARAMETERS\n"
                "of_kinetic lkt\n"
                "md_nstep 3000\n"
                f"md_csvr_tau {tau}\n",
            )
            write(point / "KPT", "K_POINTS\n")
            write(point / "STRU", "ATOMIC_SPECIES\n")
            write(point / "run_local.sh", "#!/bin/sh\nmpirun -np 12 abacus_pw_para\n")
            os.chmod(point / "run_local.sh", 0o755)
            windows.append({"label": label})
            sources.append(
                {
                    "phase": phase,
                    "label": label,
                    "source_step": 295,
                    "source_structure": str(source_structure),
                    "source_structure_sha256": sha256(source_structure),
                    "source_phase_status": f"{phase}_verified",
                    "source_minimum_nearest_neighbor_A": 2.2,
                }
            )
        write(root / phase / "manifest.json", json.dumps({"windows": windows}))
    manifest = {
        "status": "prepared",
        "target_kedf": "lkt",
        "steps_per_window": 3000,
        "mpi_ranks_per_window": 12,
        "pair_model": str(pair),
        "pair_model_sha256": sha256(pair),
        "merged_pilot_analysis": str(merged),
        "merged_pilot_analysis_sha256": sha256(merged),
        "sources": sources,
    }
    write(root / "formal_manifest.json", json.dumps(manifest))
    return root


def test_verifies_and_checksums_complete_formal_grid(tmp_path):
    root = make_grid(tmp_path)

    result = verify(
        root,
        target_kedf="lkt",
        steps=3000,
        ranks=12,
        source_step=295,
        minimum_nn=2.0,
        default_tau=5.0,
        tau_overrides={("solid", "lambda_02"): 2.0},
    )

    assert result["status"] == "verified"
    assert result["launch_authorized"] is True
    assert len(result["windows"]) == 18
    assert len((root / "INPUT_SHA256SUMS").read_text().splitlines()) == 76


def test_rejects_unexpected_existing_output(tmp_path):
    root = make_grid(tmp_path)
    (root / "liquid" / "lambda_03" / "OUT.old").mkdir()

    result = verify(
        root,
        target_kedf="lkt",
        steps=3000,
        ranks=12,
        source_step=295,
        minimum_nn=2.0,
        default_tau=5.0,
        tau_overrides={("solid", "lambda_02"): 2.0},
    )

    failed = next(
        row
        for row in result["windows"]
        if row["phase"] == "liquid" and row["label"] == "lambda_03"
    )
    assert result["status"] == "failed"
    assert failed["checks"]["no_old_out"] is False
