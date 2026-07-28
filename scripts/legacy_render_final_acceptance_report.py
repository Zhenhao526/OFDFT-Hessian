"""Render the machine-readable legacy recovery acceptance as a concise Markdown report."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def _n(value) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _metric(metrics: dict, family: str, key: str) -> str:
    return _n(metrics.get(family, {}).get(key, "n/a"))


def _table(lines: list[str], headers: list[str], rows: list[list[object]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(_n(value) for value in row) + " |" for row in rows)
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input_json.read_text())
    lines = [
        "# structures25 历史 Graphformer 恢复验收报告",
        "",
        f"机器报告：`{args.input_json.resolve()}`",
        "",
        f"最终状态：**{report['status']}**。本状态覆盖恢复、无泄漏侧车划分、双密度点科学基线、代表性密度优化以及 force/Hessian 接入准备；不表示 physical total-OFDFT force/Hessian 已实现。",
        "",
        "## 验收门",
        "",
    ]
    _table(
        lines,
        ["检查", "结果"],
        [[name, "PASS" if passed else "FAIL"] for name, passed in report["checks"].items()],
    )

    content = report["data"]["content_audit"]
    lines.extend(["## 数据完整性与划分", ""])
    _table(
        lines,
        ["物理来源", "archives", "SCF configs", "bytes", "异常"],
        [
            [
                name,
                source.get("processed"),
                source.get("scf_steps"),
                source.get("archive_bytes"),
                sum(source.get("anomaly_counts", {}).values()),
            ]
            for name, source in content["sources"].items()
        ],
    )
    safe = report["data"]["safe_splits"]["datasets"]
    _table(
        lines,
        ["虚拟数据集", "train", "val", "test", "跨划分身份重叠", "split SHA256"],
        [
            [
                name,
                item["partitions"]["train"]["entries"],
                item["partitions"]["val"]["entries"],
                item["partitions"]["test"]["entries"],
                item["cross_partition_identity_keys"],
                f"`{item['safe_split_pickle_sha256']}`",
            ]
            for name, item in safe.items()
        ],
    )
    lines.extend(
        [
            "原始 split 未修改。原始 QM9、QMUGS Bin0、组合数据集分别有 32、50、66 个 canonical-SMILES 跨划分；组合原始 split 还改变了 QM9 的历史 validation/test 身份，因此科学基线只使用按 canonical-SMILES/parent 连通组生成的侧车 split。精确几何、距离不变量、整条密度轨迹和单个构型哈希的跨/内划分重复均为 0。",
            "",
            "## 恢复验收与 golden regression",
            "",
        ]
    )
    acceptance = report["model_acceptance"]
    _table(
        lines,
        ["模型", "样本", "checkpoint SHA256", "GPU32 E/e max", "GPU32 grad max", "golden SHA256"],
        [
            [
                name,
                item["samples"],
                f"`{item['checkpoint_sha256'][0]}`",
                item["maxima"]["gpu_float32"]["energy"]["max_abs_per_electron"],
                item["maxima"]["gpu_float32"]["density_gradient"]["max_abs"],
                f"`{item['golden_report_sha256']}`",
            ]
            for name, item in acceptance["datasets"].items()
        ],
    )
    lines.extend(
        [
            "每个模型覆盖 7 个 train/val/test、原子数/元素/系数维度代表分子和 3 个密度角色，共 21 个样本；CPU float64 重复运行、GPU float32/float64、batch size>1、电子数、系数顺序和基组元数据检查均通过。无历史逐样本预测可用，因此通过所有门后冻结 CPU float64 输出为 golden。",
            "",
            "## 固定 validation 与一次性 Test 科学基线",
            "",
            "每个物理标签评估 archived SCF step 6 与最终密度。能量误差按分子聚合；density-gradient 与 difference 同时提供按系数 pooled MAE/RMSE。最终密度的 difference 标签恒为零，因此必须与非平凡的 SCF-6 组分开阅读。",
            "",
        ]
    )
    for partition in ("validation", "test"):
        lines.extend([f"### {partition}", ""])
        baselines = report["scientific_baselines"][partition]
        _table(
            lines,
            ["模型/数据域", "rows", "E MAE", "E RMSE", "grad MAE", "grad RMSE", "diff MAE", "diff RMSE"],
            [
                [
                    name,
                    item["rows"],
                    _metric(item["metrics"], "energy", "mae"),
                    _metric(item["metrics"], "energy", "rmse"),
                    _metric(item["metrics"], "projected_density_gradient", "pooled_coefficient_mae"),
                    _metric(item["metrics"], "projected_density_gradient", "pooled_coefficient_rmse"),
                    _metric(item["metrics"], "difference", "pooled_coefficient_mae"),
                    _metric(item["metrics"], "difference", "pooled_coefficient_rmse"),
                ]
                for name, item in baselines.items()
            ],
        )
        role_rows = []
        for name, item in baselines.items():
            for role, metrics in item["groups"]["density_role"].items():
                role_rows.append(
                    [
                        name,
                        role,
                        metrics["samples"],
                        _metric(metrics, "energy", "mae"),
                        _metric(metrics, "projected_density_gradient", "pooled_coefficient_mae"),
                        _metric(metrics, "difference", "pooled_coefficient_mae"),
                    ]
                )
        _table(lines, ["模型/数据域", "密度角色", "rows", "E MAE", "grad MAE", "diff MAE"], role_rows)
        source_rows = []
        for name, item in baselines.items():
            for source, metrics in item["groups"]["source"].items():
                source_rows.append(
                    [
                        name,
                        source,
                        metrics["samples"],
                        _metric(metrics, "energy", "mae"),
                        _metric(metrics, "projected_density_gradient", "pooled_coefficient_mae"),
                        _metric(metrics, "difference", "pooled_coefficient_mae"),
                    ]
                )
        _table(lines, ["模型/数据域", "物理来源", "rows", "E MAE", "grad MAE", "diff MAE"], source_rows)
        quantile_rows = []
        for name, item in baselines.items():
            for field in (
                "absolute_energy_error",
                "gradient_mae_per_coefficient",
                "difference_mae_per_coefficient",
            ):
                values = item["quantiles"]["overall"][field]
                quantile_rows.append(
                    [
                        name,
                        field,
                        values["quantiles"]["0.5"],
                        values["quantiles"]["0.9"],
                        values["quantiles"]["0.99"],
                        values["max"],
                    ]
                )
        _table(lines, ["模型/数据域", "误差", "q50", "q90", "q99", "max"], quantile_rows)
    lines.extend(
        [
            "完整机器报告还包含按物理来源、原子数和元素组成的分组，误差分位数，以及每项指标 top-100 异常样本；逐样本结果保存在各 run 的 `rows/shard_*.jsonl.gz`，其 SHA256 在报告中冻结。所有模型在 validation 完成并写入 freeze artifact 后才执行一次 Test 前向。",
            "",
            "## 模型驱动密度优化（validation 代表集）",
            "",
        ]
    )
    density = report["density_optimization"]
    _table(
        lines,
        ["模型域", "任务", "严格收敛", "成功率", "cycles median/max", "最终残差 max", "电子数误差 max", "耗时 max(s)"],
        [
            [
                name,
                group["tasks"],
                group["strict_converged"],
                group["strict_convergence_rate"],
                f"{_n(group['metrics']['cycles']['median'])}/{_n(group['metrics']['cycles']['max'])}",
                group["metrics"]["final_projected_gradient_norm"]["max"],
                group["metrics"]["optimized_electron_error"]["max"],
                group["metrics"]["elapsed_s"]["max"],
            ]
            for name, group in density["groups"].items()
        ],
    )
    lines.extend(
        [
            "QMUGS `0987810.zarr.zip` 是明确的科学非收敛：stage 1 后残差约 0.00957，stage 2 达到 10000-cycle 上限并以 0.0616 结束；日志和逐样本曲线保留。报告严格区分标签密度前向能量误差、优化后密度能量误差和系数/密度误差。",
            "",
            "## force/Hessian 接入边界",
            "",
        ]
    )
    force = report["force_hessian"]
    smoke = force["fixed_density_validation_smoke"]
    val = force["actual_force_weight_1"]["validation_selection"]
    test = force["actual_force_weight_1"]["one_shot_frozen_test100"]
    _table(
        lines,
        ["层级", "关键结果"],
        [
            ["固定密度 validation smoke", f"full Hessian {smoke['full_hessian_cases']}/{smoke['full_hessian_cases']} finite；HVP {smoke['hvp_cases']}/{smoke['hvp_cases']} finite；h={smoke['fd_displacement_bohr']} Bohr"],
            ["weight=1 validation Tier2", f"E/force/relaxed-proxy H MAE={_n(val['energy_mae'])}/{_n(val['force_component_mae'])}/{_n(val['relaxed_proxy_hessian_mae'])}；strict {val['strict_converged_points']}/{val['optimization_points']}"],
            ["frozen Test100 fixed", f"autograd-vs-FD MAE={_n(test['fixed_autograd_vs_fd_mae_mean'])}；fixed-H PBE MAE={_n(test['fixed_hessian_pbe_mae'])}；HVP/PBE-force-secant MAE={_n(test['hvp_vs_pbe_force_secant_mae'])}"],
            ["frozen Test100 relaxed proxy", f"100/100 Hessians；strict {test['relaxed_proxy_strict_points']}/{test['relaxed_proxy_optimization_points']}；MAE={_n(test['relaxed_proxy_hessian_mae'])}"],
        ],
    )
    lines.extend(
        [
            "恢复的历史 Graphformer 只预测能量、密度梯度和 density difference，不是 force/Hessian 模型。固定密度结果只微分 learned scalar energy，缺少密度响应、classical integral/nuclear 与 Pulay 坐标导数；密度松弛结果是 incomplete-derived-force proxy，也不是 physical total-OFDFT Hessian。单位为 Hartree、Bohr、Hartree/Bohr、Hartree/Bohr²。",
            "",
            "## 保守 total-OFDFT 导数实施门",
            "",
        ]
    )
    plan = report["total_derivative"]["local_readiness"]["conservative_total_ofdft_plan"]
    for stage in plan:
        lines.append(f"{stage['stage']}. **{stage['name']}** — {stage['gate']}")
        for work in stage["work"]:
            lines.append(f"   - {work}")
    lines.extend(
        [
            "",
            f"本地候选组件测试：{report['total_derivative']['local_unit_tests']['tests']} passed，errors={report['total_derivative']['local_unit_tests']['errors']}，failures={report['total_derivative']['local_unit_tests']['failures']}。远端恢复快照尚未部署这些接口；在 clean code-release audit 前保持 `planned_not_deployed_on_recovery_server`。在中心差分、闭合回路、KKT/HVP、平移/转动及单位门全部通过前，不给出物理振动结论。",
            "",
            "## 失败任务与计算成本",
            "",
        ]
    )
    _table(
        lines,
        ["任务", "状态", "原因", "替代/结论"],
        [[item["id"], item["state"], item["cause"], item["resolution"]] for item in report["failed_attempts"]["attempts"]],
    )
    lines.extend(
        [
            "上述任务均标记 `used_as_scientific_result=false`，保留日志 SHA256。baseline 报告中的 `cost` 给出各 shard 设备秒总和、批次数与并行 wall-time 估计；密度优化与 Test100 proxy 另存逐任务 cycles/耗时。",
            "",
            "## 可复现命令与路径策略",
            "",
            "所有入口只接受 `DFT_DATA=/scratch/xzh/dft/data`、`DFT_MODELS=/scratch/xzh/dft/models` 或等价显式参数。`hparams_resolved.yaml` 中 `/export/scratch/ialgroup` 仅作为 provenance 记录；若成为有效路径，程序立即报错。代表性 baseline shard 命令如下，其余 shard 只改变 shard index：",
            "",
        ]
    )
    for partition in ("validation", "test"):
        for name, item in report["scientific_baselines"][partition].items():
            command = item["execution"].get("command")
            if command:
                lines.extend([f"`{partition}/{name}`", "", "```bash", shlex.join(map(str, command)), "```", ""])
    lines.extend(
        [
            "## 结论与下一步",
            "",
            "1. 恢复、数据审计、侧车无泄漏划分和历史模型验收已冻结；不需要因工程恢复而重训。",
            "2. 先处理 QMUGS 优化 hard case，并仅在 validation 上冻结任何 solver 策略；Test 不用于调参。",
            "3. 将本地 tensor-energy/integral/Pulay/KKT 候选通过 clean release 部署远端，按 stage 0–5 逐门验证。",
            "4. 保持 fixed-density 与 incomplete-derived-force proxy 的命名边界；conservative total force 通过完全松弛总能量中心差分和闭合回路后，才进入 implicit Hessian 与物理振动。",
            "5. 只有上述 validation 门冻结后才决定是否启动全量重训；当前没有科学依据立即重训历史 Graphformer。",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
