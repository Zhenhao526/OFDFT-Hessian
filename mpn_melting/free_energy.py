from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from .parser import parse_text
from .trajectory import Matrix, Vector, lattice_volume, parse_md_dump


KB_EV_PER_K = 8.617333262145e-5


@dataclass(frozen=True)
class FreeEnergyFrame:
    phase: str
    source_run: str
    source_index: int
    md_step: int
    temperature_k: float
    potential_energy_ev: float
    lattice_angstrom: Matrix
    positions_angstrom: Sequence[Vector]
    forces_ev_per_angstrom: Sequence[Vector]

    @property
    def natoms(self) -> int:
        return len(self.positions_angstrom)

    def to_json_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["natoms"] = self.natoms
        value["volume_angstrom3"] = lattice_volume(self.lattice_angstrom)
        value["potential_energy_ev_per_atom"] = self.potential_energy_ev / self.natoms
        value["force_rms_ev_per_angstrom"] = force_rms(self.forces_ev_per_angstrom)
        return value


def force_rms(forces: Sequence[Vector]) -> float:
    if not forces:
        return 0.0
    return math.sqrt(
        sum(component * component for force in forces for component in force)
        / (3 * len(forces))
    )


def exponential_free_energy_difference(
    energy_differences_ev: Sequence[float], temperature_k: float
) -> Dict[str, float]:
    """Return -kT log <exp(-beta dU)> and exponential-weight ESS."""
    if not energy_differences_ev:
        raise ValueError("at least one energy difference is required")
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    beta = 1.0 / (KB_EV_PER_K * temperature_k)
    exponents = [-beta * value for value in energy_differences_ev]
    shift = max(exponents)
    weights = [math.exp(value - shift) for value in exponents]
    weight_sum = sum(weights)
    log_average = shift + math.log(weight_sum / len(weights))
    effective_sample_size = weight_sum * weight_sum / sum(value * value for value in weights)
    return {
        "delta_f_ev": -log_average / beta,
        "effective_sample_size": effective_sample_size,
        "effective_sample_fraction": effective_sample_size / len(weights),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_run_files(run_dir: Path) -> Tuple[Path, Path]:
    dumps = sorted(run_dir.rglob("MD_dump"))
    logs = sorted(run_dir.rglob("running_md.log"))
    if len(dumps) != 1:
        raise ValueError(f"expected one MD_dump below {run_dir}, found {len(dumps)}")
    if len(logs) != 1:
        raise ValueError(f"expected one running_md.log below {run_dir}, found {len(logs)}")
    return dumps[0], logs[0]


def load_run_frames(run_dir: Path, phase: str) -> Tuple[List[FreeEnergyFrame], Dict[str, object]]:
    if not phase.strip():
        raise ValueError("phase must not be empty")
    dump_path, log_path = find_run_files(run_dir)
    md_frames = parse_md_dump(dump_path)
    series = parse_text(log_path.read_text(encoding="utf-8", errors="ignore"))
    nobservables = len(series.md_potential_energies)
    if len(md_frames) != nobservables or len(series.temperatures) != nobservables:
        raise ValueError(
            "trajectory/observable mismatch for "
            f"{run_dir}: frames={len(md_frames)}, potentials={nobservables}, "
            f"temperatures={len(series.temperatures)}"
        )
    if not md_frames:
        raise ValueError(f"no MD frames found below {run_dir}")

    natoms = len(md_frames[0].positions)
    frames: List[FreeEnergyFrame] = []
    for index, (frame, potential, temperature) in enumerate(
        zip(md_frames, series.md_potential_energies, series.temperatures)
    ):
        if len(frame.positions) != natoms:
            raise ValueError(f"atom count changes at frame {index}: {len(frame.positions)} != {natoms}")
        if frame.forces is None or len(frame.forces) != natoms:
            raise ValueError(f"missing forces at frame {index} in {dump_path}")
        frames.append(
            FreeEnergyFrame(
                phase=phase,
                source_run=str(run_dir.resolve()),
                source_index=index,
                md_step=frame.step,
                temperature_k=temperature,
                potential_energy_ev=potential,
                lattice_angstrom=frame.lattice,
                positions_angstrom=frame.positions,
                forces_ev_per_angstrom=frame.forces,
            )
        )
    source = {
        "phase": phase,
        "run_dir": str(run_dir.resolve()),
        "natoms": natoms,
        "available_frames": len(frames),
        "md_dump": str(dump_path.resolve()),
        "md_dump_sha256": sha256_file(dump_path),
        "running_md_log": str(log_path.resolve()),
        "running_md_log_sha256": sha256_file(log_path),
    }
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target_kedf = str(
            metadata.get("target_kedf")
            or metadata.get("abacus", {}).get("of_kinetic")
            or ""
        ).lower()
        if target_kedf:
            source["target_kedf"] = target_kedf
            source["metadata"] = str(metadata_path.resolve())
            source["metadata_sha256"] = sha256_file(metadata_path)
    return frames, source


def select_frames(
    frames: Sequence[FreeEnergyFrame], discard_fraction: float = 0.5, stride: int = 1
) -> List[FreeEnergyFrame]:
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError("discard_fraction must be in [0, 1)")
    if stride < 1:
        raise ValueError("stride must be at least one")
    start = int(len(frames) * discard_fraction)
    selected = list(frames[start::stride])
    if not selected:
        raise ValueError("frame selection is empty")
    return selected


def write_dataset(
    runs: Iterable[Tuple[str, Path]],
    output_dir: Path,
    discard_fraction: float = 0.5,
    stride: int = 1,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_path = output_dir / "frames.jsonl"
    sources: List[Dict[str, object]] = []
    phase_counts: Dict[str, int] = {}
    total_frames = 0
    natoms_values = set()
    target_kedfs = set()

    with frames_path.open("w", encoding="utf-8") as handle:
        for phase, run_dir in runs:
            available, source = load_run_frames(run_dir, phase)
            selected = select_frames(available, discard_fraction, stride)
            source["selected_frames"] = len(selected)
            source["selected_source_indices"] = [frame.source_index for frame in selected]
            sources.append(source)
            phase_counts[phase] = phase_counts.get(phase, 0) + len(selected)
            natoms_values.add(source["natoms"])
            if source.get("target_kedf"):
                target_kedfs.add(str(source["target_kedf"]))
            for frame in selected:
                handle.write(json.dumps(frame.to_json_dict(), separators=(",", ":")) + "\n")
            total_frames += len(selected)

    if len(target_kedfs) > 1:
        raise ValueError(
            "dataset sources use different KEDFs: "
            + ", ".join(sorted(target_kedfs))
        )
    manifest: Dict[str, object] = {
        "schema": "mpn-free-energy-dataset-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "units": {
            "length": "angstrom",
            "energy": "eV",
            "force": "eV/angstrom",
            "temperature": "K",
        },
        "selection": {"discard_fraction": discard_fraction, "stride": stride},
        "total_frames": total_frames,
        "phase_counts": phase_counts,
        "natoms_values": sorted(natoms_values),
        "sources": sources,
        "frames_file": frames_path.name,
        "frames_sha256": sha256_file(frames_path),
    }
    if target_kedfs:
        manifest["target_kedf"] = target_kedfs.pop()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
