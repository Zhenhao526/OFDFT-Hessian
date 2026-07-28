"""Static graph-cut audit and staged conservative total-OFDFT derivative plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TARGETS = (
    "mldft/ofdft/energies.py",
    "mldft/ofdft/functional_factory.py",
    "mldft/ofdft/density_optimization.py",
    "mldft/ofdft/optimizer.py",
    "mldft/ofdft/run_density_optimization.py",
    "mldft/ml/data/components/convert_transforms.py",
    "mldft/ml/data/components/basis_transforms.py",
    "mldft/ml/models/mldft_module.py",
)
PATTERNS = {
    "tensor_to_python_item": re.compile(r"\.item\s*\("),
    "tensor_detach": re.compile(r"\.detach\s*\("),
    "tensor_to_cpu": re.compile(r"\.cpu\s*\("),
    "tensor_to_numpy": re.compile(r"\.numpy\s*\("),
    "python_float_cast": re.compile(r"(?<![A-Za-z_])float\s*\("),
    "numpy_operation": re.compile(r"\bnp\."),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    root = args.code_root.resolve()
    findings = []
    files = {}
    for relative in TARGETS:
        path = root / relative
        if not path.is_file():
            findings.append({"file": relative, "kind": "missing_target", "line": None})
            continue
        text = path.read_text(errors="replace")
        files[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for number, line in enumerate(text.splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(
                        {
                            "file": relative,
                            "line": number,
                            "kind": kind,
                            "text": line.strip()[:300],
                            "disposition": (
                                "must be removed or moved behind a detached reporting boundary "
                                "if reached by total-energy coordinate differentiation"
                            ),
                        }
                    )

    functional_text = (root / "mldft/ofdft/functional_factory.py").read_text()
    energies_text = (root / "mldft/ofdft/energies.py").read_text()
    deployment = {
        "TensorEnergies": "class TensorEnergies" in energies_text,
        "evaluate_tensor_functional": "def evaluate_tensor_functional" in functional_text,
        "construct_tensor": "def construct_tensor" in functional_text,
        "geometry_integrals_module": (root / "mldft/ofdft/geometry_integrals.py").is_file(),
        "conservative_force_module": (root / "mldft/ofdft/conservative_force.py").is_file(),
        "implicit_response_module": (root / "mldft/ofdft/implicit_response.py").is_file(),
    }
    counts = {}
    for row in findings:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1

    report = {
        "status": "audit_complete",
        "scope": (
            "Static audit only. A finding is a candidate graph cut, not proof that the line is "
            "executed by the total-energy coordinate derivative. Dynamic graph tests are required."
        ),
        "code_root": str(root),
        "files": files,
        "deployment": deployment,
        "finding_counts": counts,
        "findings": findings,
        "conservative_total_ofdft_plan": [
            {
                "stage": 0,
                "name": "freeze interfaces and reporting boundaries",
                "work": [
                    "Keep legacy float/Energies APIs for logs only and add an explicit tensor-returning scalar-energy API.",
                    "Make .item(), float(), NumPy, detach and CPU transfers fail dynamic graph tests when reached before the reporting boundary.",
                    "Version and hash all integral providers, basis metadata, checkpoint and validation geometry lists.",
                ],
                "gate": "Tensor scalar energy retains grad_fn and passes first/second coordinate derivative smoke tests in float64.",
            },
            {
                "stage": 1,
                "name": "fixed-coefficient total scalar energy",
                "work": [
                    "Assemble learned kinetic+XC, Hartree 0.5*c^T*J*c, electron-nuclear c^T*v_ext and nuclear-nuclear repulsion as tensors.",
                    "Keep electron normalization q(R)^T c=N in tensor form and validate coefficient/basis ordering.",
                ],
                "gate": "Every component and the sum agree with central finite differences over a validation-only small-molecule set and several step sizes.",
            },
            {
                "stage": 2,
                "name": "moving-integral and Pulay response",
                "work": [
                    "Provide coordinate JVP/VJP for Coulomb J(R), nuclear attraction v_ext(R), overlap/metric S(R), dual integrals q(R), transformations and E_nn(R).",
                    "Use analytic derivatives or audited custom autograd; if finite-difference-backed, expose truncation/error controls and adjoint tests.",
                    "Include moving auxiliary basis/local frames and all Pulay terms rather than treating cached tensors as constants.",
                ],
                "gate": "Component VJP/JVP adjoint consistency plus full fixed-density force agreement with total scalar-energy central differences.",
            },
            {
                "stage": 3,
                "name": "constrained relaxed total force",
                "work": [
                    "Solve stationarity of L(c,mu,R)=E(c,R)+mu*(q(R)^T*c-N) in the electron-number tangent space.",
                    "Use the envelope derivative -partial_R L at a converged density and include q(R), normalization and Pulay response.",
                    "Separate solver failure, nonstationary residual and derivative failure in reports.",
                ],
                "gate": "Relaxed force matches fully reoptimized E*(R+h)-E*(R-h) and closed-loop work is step-convergent toward zero on validation geometries.",
            },
            {
                "stage": 4,
                "name": "implicit density response and Hessian-vector product",
                "work": [
                    "Build the KKT/tangent operator [L_cc q; q^T 0] and solve its directional response with dense reference then MINRES/PCG-compatible blocks.",
                    "Combine direct coordinate curvature, mixed c-R blocks, q(R) derivatives and implicit dc/dR without unrolling optimizer branches.",
                    "Check HVP against finite differences of the conservative total force and verify Hessian symmetry.",
                ],
                "gate": "KKT residual, tangent electron constraint, HVP finite-difference agreement and symmetry all pass preregistered validation tolerances.",
            },
            {
                "stage": 5,
                "name": "physical interpretation gate",
                "work": [
                    "Freeze implementation and validation tolerances before one test evaluation.",
                    "Only after translation/rotation response, conservation, units and mass weighting pass may physical vibrational quantities be reported.",
                ],
                "gate": "Until all previous gates pass, label outputs as fixed-density or incomplete-derived-force proxies, never physical total-OFDFT Hessians.",
            },
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "finding_counts": counts, "deployment": deployment}, sort_keys=True))


if __name__ == "__main__":
    main()
