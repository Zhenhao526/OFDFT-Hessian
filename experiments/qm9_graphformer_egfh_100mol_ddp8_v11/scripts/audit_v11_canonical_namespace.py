#!/usr/bin/env python3
"""Static fail-closed audit of v11 versus canonical namespace consumers."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chain(node: ast.AST) -> list[str] | None:
    values: list[str] = []
    while isinstance(node, ast.Attribute):
        values.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        values.append(node.id)
        return list(reversed(values))
    return None


def module_functions(tree: ast.Module) -> dict[str, ast.AST]:
    functions: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions[f"{node.name}.{child.name}"] = child
    return functions


def local_calls(node: ast.AST, available: set[str]) -> set[str]:
    calls: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name) and child.func.id in available:
            calls.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            value = chain(child.func)
            if value and len(value) == 2 and value[0] == "self":
                suffix = f".{value[1]}"
                calls.update(name for name in available if name.endswith(suffix))
    return calls


def reachable(functions: dict[str, ast.AST], seeds: set[str]) -> set[str]:
    found = {seed for seed in seeds if seed in functions}
    changed = True
    while changed:
        changed = False
        for name in list(found):
            for called in local_calls(functions[name], set(functions)):
                if called not in found:
                    found.add(called)
                    changed = True
    return found


def namespace_fields(node: ast.AST, roots: set[tuple[str, ...]]) -> tuple[set[str], set[str]]:
    required: set[str] = set()
    optional: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            value = chain(child)
            if value and len(value) >= 2:
                prefix = tuple(value[:-1])
                if prefix in roots:
                    required.add(value[-1])
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "getattr"
            and len(child.args) >= 2
            and isinstance(child.args[1], ast.Constant)
            and isinstance(child.args[1].value, str)
        ):
            value = chain(child.args[0])
            if value and tuple(value) in roots:
                optional.add(child.args[1].value)
    return required, optional


def trainer_fields(tree: ast.Module) -> tuple[set[str], set[str]]:
    parser_fields: set[str] = set()
    injected_fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            option = next((arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")), None)
            if option:
                destination = next((kw.value.value for kw in node.keywords if kw.arg == "dest" and isinstance(kw.value, ast.Constant)), None)
                parser_fields.add(str(destination or option[2:].replace("-", "_")))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "density_namespace":
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    for key in child.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            injected_fields.add(key.value)
    return parser_fields, injected_fields


def analyze_module(path: Path, seeds: set[str], roots: set[tuple[str, ...]], include_class: str | None = None) -> dict[str, Any]:
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = module_functions(tree)
    active = reachable(functions, seeds)
    if include_class:
        active.update(name for name in functions if name.startswith(include_class + "."))
    required: set[str] = set()
    optional: set[str] = set()
    for name in active:
        found_required, found_optional = namespace_fields(functions[name], roots)
        required.update(found_required)
        optional.update(found_optional)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "seeds": sorted(seeds),
        "reachable_functions": sorted(active),
        "required_fields": sorted(required),
        "optional_fields": sorted(optional - required),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--force-audit", type=Path, required=True)
    parser.add_argument("--context-loader", type=Path, required=True)
    parser.add_argument("--v10-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trainer_tree = ast.parse(args.trainer.read_text(), filename=str(args.trainer))
    parser_fields, injected_fields = trainer_fields(trainer_tree)
    effective = parser_fields | injected_fields
    core_seeds = {
        node.attr
        for node in ast.walk(trainer_tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "core"
    }
    canonical = analyze_module(
        args.canonical,
        core_seeds,
        {("density_args",), ("cache", "args"), ("self", "args")},
        include_class="IntegralBundleCache",
    )
    force = analyze_module(
        args.force_audit,
        {"_evaluate_point", "_initialization"},
        {("args",)},
    )
    context = analyze_module(
        args.context_loader,
        {"_load_context", "_parse_run"},
        {("args",)},
    )
    required = set(canonical["required_fields"]) | set(force["required_fields"]) | set(context["required_fields"])
    optional = set(canonical["optional_fields"]) | set(force["optional_fields"]) | set(context["optional_fields"])
    missing = sorted(required - effective)
    v10 = json.loads(args.v10_summary.read_text())
    v10_values = {name: v10.get(name, "__NOT_RECORDED__") for name in sorted(required)}
    report = {
        "artifact_id": "qm9_v11_canonical_namespace_static_audit_v1",
        "status": "pass" if not missing else "fail",
        "trainer": {"path": str(args.trainer), "sha256": sha256(args.trainer)},
        "trainer_parser_fields": sorted(parser_fields),
        "trainer_density_injected_fields": sorted(injected_fields),
        "effective_namespace_fields": sorted(effective),
        "canonical_modules": [canonical, force, context],
        "required_fields_union": sorted(required),
        "optional_fields_union": sorted(optional - required),
        "missing_required_fields": missing,
        "v10_recorded_values": v10_values,
        "note": "Static over-approximation: class methods are included fail-closed even when a legacy branch is inactive.",
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
