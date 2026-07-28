#!/usr/bin/env python3
"""Aggregate HORM pilot label checks and write a failure CSV/report."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import zarr


ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--ks-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--failure-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--nthreads", type=int, required=True)
    parser.add_argument("--pyscf-max-memory", type=int, required=True)
    parser.add_argument("--ks-command", required=True)
    parser.add_argument("--labelgen-command", required=True)
    return parser.parse_args()


def to_numpy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def composition_from_z(z: np.ndarray) -> str:
    counts = Counter(int(x) for x in z.tolist())
    return "".join(f"{ELEMENTS.get(k, str(k))}{counts[k]}" for k in sorted(counts))


def load_raw_metadata(env: lmdb.Environment, sample_id: int) -> dict[str, Any]:
    with env.begin(write=False) as txn:
        value = txn.get(str(sample_id).encode())
    if value is None:
        return {"natoms": "", "composition": ""}
    data = pickle.loads(value)
    z = np.asarray(to_numpy(data.charges), dtype=np.int64)
    natoms = int(data.natoms.item()) if hasattr(data.natoms, "item") else int(data.natoms)
    return {"natoms": natoms, "composition": composition_from_z(z)}


def walk_arrays(group: zarr.Group, prefix: str = "") -> list[tuple[str, zarr.Array]]:
    arrays = []
    for key, value in sorted(group.items()):
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, zarr.Array):
            arrays.append((path, value))
        else:
            arrays.extend(walk_arrays(value, path))
    return arrays


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float64)
        if data.size == 0:
            return
        self.count += int(data.size)
        self.sum += float(data.sum())
        self.sumsq += float(np.square(data).sum())
        self.min = min(self.min, float(data.min()))
        self.max = max(self.max, float(data.max()))

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "std": var**0.5,
        }


def add_range(ranges: dict[str, dict[str, float]], key: str, data: np.ndarray) -> None:
    if data.size == 0 or not np.issubdtype(data.dtype, np.number) or data.dtype == np.bool_:
        return
    value_min = float(np.nanmin(data))
    value_max = float(np.nanmax(data))
    if key not in ranges:
        ranges[key] = {"min": value_min, "max": value_max}
    else:
        ranges[key]["min"] = min(ranges[key]["min"], value_min)
        ranges[key]["max"] = max(ranges[key]["max"], value_max)


def format_counter(counter: Counter) -> str:
    return ", ".join(f"{k}: {v}" for k, v in sorted(counter.items()))


def markdown_table(rows: list[tuple[Any, ...]], headers: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sample_ids = list(range(args.start, args.start + args.n))

    env = lmdb.open(
        args.archive.as_posix(),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
    )

    failures = []
    successful = []
    atom_counts = []
    compositions = Counter()
    coeff_dims = []
    scf_steps = []
    nan_count = 0
    inf_count = 0
    arrays_with_nan = []
    arrays_with_inf = []
    energy_ranges: dict[str, dict[str, float]] = {}
    grad_stats = {"grad_kin_plus_xc": RunningStats(), "grad_tot": RunningStats()}
    ref_forces_count = 0
    ref_hessian_count = 0
    pbe_derivatives_count = 0
    checked_labels = 0

    for sample_id in sample_ids:
        raw = load_raw_metadata(env, sample_id)
        chk_path = args.ks_dir / f"horm_{sample_id:07}.chk"
        label_path = args.label_dir / f"{sample_id:07}.zarr.zip"
        if not chk_path.exists():
            failures.append(
                {
                    "sample_id": sample_id,
                    "molecule_id": sample_id,
                    "stage": "KSDFT",
                    "error_type": "MissingChk",
                    "message": f"Missing {chk_path}",
                    **raw,
                }
            )
            continue
        if not label_path.exists():
            failures.append(
                {
                    "sample_id": sample_id,
                    "molecule_id": sample_id,
                    "stage": "labelgen",
                    "error_type": "MissingLabel",
                    "message": f"Missing {label_path}",
                    **raw,
                }
            )
            continue
        try:
            root = zarr.open(label_path, mode="r")
            arrays = walk_arrays(root)
            for key, array in arrays:
                if not np.issubdtype(array.dtype, np.number):
                    continue
                data = np.asarray(array)
                n_nan = int(np.isnan(data).sum()) if np.issubdtype(data.dtype, np.floating) else 0
                n_inf = int(np.isinf(data).sum()) if np.issubdtype(data.dtype, np.floating) else 0
                nan_count += n_nan
                inf_count += n_inf
                if n_nan:
                    arrays_with_nan.append(f"{sample_id}:{key}")
                if n_inf:
                    arrays_with_inf.append(f"{sample_id}:{key}")
            z = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
            atom_counts.append(int(z.size))
            compositions[composition_from_z(z)] += 1
            coeffs = root["of_labels/spatial/coeffs"]
            coeff_dims.append(int(coeffs.shape[1]))
            scf_steps.append(int(root["of_labels/n_scf_steps"][()]))
            for group_name in ("ks_labels/energies", "of_labels/energies"):
                group = root[group_name]
                for key, array in group.items():
                    add_range(energy_ranges, f"{group_name}/{key}", np.asarray(array))
            for key in grad_stats:
                grad_stats[key].update(np.asarray(root[f"of_labels/spatial/{key}"]))
            if "metadata/reference/forces" in root:
                ref_forces_count += 1
            if "metadata/reference/hessian" in root:
                ref_hessian_count += 1
            if "metadata/pbe_derivatives" in root:
                pbe_derivatives_count += 1
            successful.append(sample_id)
            checked_labels += 1
        except Exception as exc:
            failures.append(
                {
                    "sample_id": sample_id,
                    "molecule_id": sample_id,
                    "stage": "check",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:300],
                    **raw,
                }
            )

    env.close()

    args.failure_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.failure_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "molecule_id",
                "stage",
                "error_type",
                "message",
                "natoms",
                "composition",
            ],
        )
        writer.writeheader()
        writer.writerows(failures)

    summary = {
        "expected_count": len(sample_ids),
        "success_count": len(successful),
        "failure_count": len(failures),
        "successful_ids": successful,
        "failed_ids": [row["sample_id"] for row in failures],
        "nan_count": nan_count,
        "inf_count": inf_count,
        "arrays_with_nan": arrays_with_nan,
        "arrays_with_inf": arrays_with_inf,
        "atom_count_min": min(atom_counts) if atom_counts else None,
        "atom_count_max": max(atom_counts) if atom_counts else None,
        "atom_count_distribution": dict(Counter(atom_counts)),
        "composition_distribution": dict(compositions),
        "coeff_dim_min": min(coeff_dims) if coeff_dims else None,
        "coeff_dim_max": max(coeff_dims) if coeff_dims else None,
        "coeff_dim_distribution": dict(Counter(coeff_dims)),
        "scf_steps_min": min(scf_steps) if scf_steps else None,
        "scf_steps_max": max(scf_steps) if scf_steps else None,
        "scf_steps_distribution": dict(Counter(scf_steps)),
        "energy_ranges": energy_ranges,
        "grad_kin_plus_xc": grad_stats["grad_kin_plus_xc"].as_dict(),
        "grad_tot": grad_stats["grad_tot"].as_dict(),
        "reference_forces_count": ref_forces_count,
        "reference_hessian_count": ref_hessian_count,
        "pbe_derivatives_count": pbe_derivatives_count,
        "label_dir": args.label_dir.as_posix(),
        "ks_dir": args.ks_dir.as_posix(),
        "archive": args.archive.as_posix(),
        "failure_csv": args.failure_csv.as_posix(),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    atom_rows = sorted(Counter(atom_counts).items())
    coeff_rows = sorted(Counter(coeff_dims).items())
    scf_rows = sorted(Counter(scf_steps).items())
    comp_rows = sorted(compositions.items())
    energy_rows = [(key, f"{val['min']:.12g}", f"{val['max']:.12g}") for key, val in sorted(energy_ranges.items())]

    max_grad_abs = max(
        abs(summary["grad_kin_plus_xc"]["min"]),
        abs(summary["grad_kin_plus_xc"]["max"]),
        abs(summary["grad_tot"]["min"]),
        abs(summary["grad_tot"]["max"]),
    )
    abnormal_gradient = max_grad_abs > 1e6
    recommend_500 = len(failures) == 0 and nan_count == 0 and inf_count == 0 and not abnormal_gradient
    recommend_full = False

    report = f"""# HORM 50 样本 Pilot 报告

更新时间：2026-07-01

## 运行配置

输入数据：`{args.archive}`

KSDFT 输出：`{args.ks_dir}`

标签输出：`{args.label_dir}`

NPROC={args.nproc}, NTHREADS={args.nthreads}, PYSCF_MAX_MEMORY={args.pyscf_max_memory} MB

KSDFT 命令：

```bash
{args.ks_command}
```

labelgen 命令：

```bash
{args.labelgen_command}
```

## 成功/失败统计

- 期望样本数：{len(sample_ids)}
- 成功标签数：{len(successful)}
- 失败数：{len(failures)}
- failure CSV：`{args.failure_csv}`
- 统计 JSON：`{args.summary_json}`

## 数值健康检查

- NaN 数量：{nan_count}
- Inf 数量：{inf_count}
- `metadata/reference/forces` 覆盖：{ref_forces_count}/{checked_labels}
- `metadata/reference/hessian` 覆盖：{ref_hessian_count}/{checked_labels}
- PBE force/Hessian 重算标签：{pbe_derivatives_count}/{checked_labels}（本 pilot 默认关闭）

## 元素组成统计

{markdown_table(comp_rows, ("composition", "count")) if comp_rows else "无成功样本。"}

## 原子数分布

范围：{summary["atom_count_min"]} - {summary["atom_count_max"]}

{markdown_table(atom_rows, ("natoms", "count")) if atom_rows else "无成功样本。"}

## Density Coefficient 维度分布

范围：{summary["coeff_dim_min"]} - {summary["coeff_dim_max"]}

{markdown_table(coeff_rows, ("n_coeffs", "count")) if coeff_rows else "无成功样本。"}

## SCF Iteration 分布

范围：{summary["scf_steps_min"]} - {summary["scf_steps_max"]}

{markdown_table(scf_rows, ("n_scf_steps", "count")) if scf_rows else "无成功样本。"}

## Energy Label 范围

{markdown_table(energy_rows, ("label", "min", "max")) if energy_rows else "无成功样本。"}

## Gradient 统计

`grad_kin_plus_xc`：

```json
{json.dumps(summary["grad_kin_plus_xc"], indent=2)}
```

`grad_tot`：

```json
{json.dumps(summary["grad_tot"], indent=2)}
```

异常大梯度判断：{"发现，max_abs > 1e6" if abnormal_gradient else "未发现，max_abs <= 1e6"}。

## 失败样本

{"无失败样本。" if not failures else "失败样本见 failure CSV。"}

## 扩大建议

- 建议进入 500/1000 样本阶段：{"是" if recommend_500 else "否"}
- 建议直接全量跑：{"是" if recommend_full else "否"}

当前 50 样本 pilot 成功率、NaN/Inf 检查和 reference metadata 覆盖都支持进入 500/1000 样本阶段；但不建议直接全量跑，应先用 500/1000 样本验证长尾原子数、SCF 收敛、labelgen 内存和存储增长。

## DFT Level 警告

HORM archive 原始 force/Hessian 来源标注为 `omegaB97x/6-31G(d)`。本流程新生成的 STRUCTURES25/M-OFDFT 标签是 `PBE/6-31G(2df,p)`。`metadata/reference/*` 中的原始 HORM energy/force/Hessian 不能和 PBE 标签混合作为同一训练目标。
"""
    args.report.write_text(report)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
