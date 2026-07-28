"""Throughput and coarse step timing callback for full-scale calibration."""

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback


class ThroughputMonitor(Callback):
    """Record training throughput and coarse timing to CSV/JSON.

    The callback is intentionally lightweight and rank-zero-only for file output. In DDP each rank
    sees a per-rank batch, so global samples are estimated as ``local_batch_size * world_size``.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        log_every_n_steps: int = 10,
        warmup_steps: int = 1,
        estimate_num_samples: int | None = None,
        estimate_epochs: Iterable[int] | None = None,
        target_num_devices: int = 8,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.log_every_n_steps = int(log_every_n_steps)
        self.warmup_steps = int(warmup_steps)
        self.estimate_num_samples = estimate_num_samples
        self.estimate_epochs = list(estimate_epochs or [10, 20])
        self.target_num_devices = int(target_num_devices)

        self.csv_path: Path | None = None
        self.json_path: Path | None = None
        self.rows: list[dict[str, Any]] = []
        self._fit_start_time = 0.0
        self._last_batch_end_time = 0.0
        self._batch_start_time = 0.0
        self._backward_start_time: float | None = None
        self._optimizer_start_time: float | None = None
        self._interval_start_time = 0.0
        self._interval_samples = 0
        self._interval_batches = 0
        self._total_samples = 0
        self._total_batches = 0
        self._current_data_time = 0.0
        self._current_backward_time = 0.0
        self._current_optimizer_time = 0.0
        self._last_world_size = 1
        self._last_local_batch_size = 1
        self._last_accumulate_grad_batches = 1
        self._last_timing: dict[str, float] = {}

    @staticmethod
    def _now() -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _local_batch_size(batch: Any) -> int:
        if hasattr(batch, "batch_size"):
            return int(batch.batch_size)
        try:
            return len(batch)
        except TypeError:
            return 1

    @staticmethod
    def _max_cuda_memory_mb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated() / 1024**2)

    def _world_size(self, trainer: pl.Trainer) -> int:
        strategy = getattr(trainer, "strategy", None)
        world_size = getattr(strategy, "world_size", None)
        if world_size is None:
            world_size = getattr(trainer, "world_size", 1)
        return int(world_size or 1)

    def _accumulate_grad_batches(self, trainer: pl.Trainer) -> int:
        value = getattr(trainer, "accumulate_grad_batches", 1)
        if isinstance(value, dict):
            return int(next(iter(value.values())))
        return int(value or 1)

    def _write_csv_header(self) -> None:
        if self.csv_path is None or self.rows:
            return
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            writer.writeheader()

    @staticmethod
    def _csv_fields() -> list[str]:
        return [
            "global_step",
            "total_batches",
            "world_size",
            "per_gpu_batch_size",
            "accumulate_grad_batches",
            "effective_global_batch_size",
            "interval_batches",
            "interval_samples",
            "samples_per_sec",
            "steps_per_sec",
            "data_loading_s_per_batch",
            "net_forward_s_per_batch",
            "density_gradient_autograd_s_per_batch",
            "force_autograd_s_per_batch",
            "hvp_autograd_s_per_batch",
            "backward_s_per_batch",
            "optimizer_s_per_batch",
            "peak_gpu_memory_mb",
            "estimated_epoch_seconds",
        ]

    def _append_csv_row(self, row: dict[str, Any]) -> None:
        if self.csv_path is None:
            return
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            writer.writerow({key: row.get(key) for key in self._csv_fields()})

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._fit_start_time = self._now()
        self._last_batch_end_time = self._fit_start_time
        self._interval_start_time = self._fit_start_time
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if not trainer.is_global_zero:
            return

        output_dir = self.output_dir or Path(trainer.default_root_dir) / "throughput"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = output_dir / "throughput.csv"
        self.json_path = output_dir / "throughput_summary.json"
        self._write_csv_header()

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        now = self._now()
        self._current_data_time = max(0.0, now - self._last_batch_end_time)
        self._batch_start_time = now
        self._current_backward_time = 0.0
        self._current_optimizer_time = 0.0

    def on_before_backward(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, loss: torch.Tensor
    ) -> None:
        self._backward_start_time = self._now()

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._backward_start_time is not None:
            self._current_backward_time += max(0.0, self._now() - self._backward_start_time)
            self._backward_start_time = None

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self._optimizer_start_time = self._now()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        now = self._now()
        if self._optimizer_start_time is not None:
            self._current_optimizer_time += max(0.0, now - self._optimizer_start_time)
            self._optimizer_start_time = None

        world_size = self._world_size(trainer)
        local_batch_size = self._local_batch_size(batch)
        self._last_world_size = world_size
        self._last_local_batch_size = local_batch_size
        self._last_accumulate_grad_batches = self._accumulate_grad_batches(trainer)
        self._last_timing = getattr(pl_module, "_last_forward_timing", {}) or {}
        global_samples = local_batch_size * world_size
        self._total_batches += 1
        self._total_samples += global_samples

        if self._total_batches > self.warmup_steps:
            self._interval_batches += 1
            self._interval_samples += global_samples

        self._last_batch_end_time = now

        should_log = (
            self._total_batches > self.warmup_steps
            and self._interval_batches > 0
            and self._interval_batches % self.log_every_n_steps == 0
        )
        if not should_log:
            return

        row = self._build_interval_row(trainer=trainer, now=now)
        self.rows.append(row)
        if trainer.is_global_zero:
            self._append_csv_row(row)

        self._interval_start_time = now
        self._interval_samples = 0
        self._interval_batches = 0

    def _build_interval_row(self, trainer: pl.Trainer, now: float) -> dict[str, Any]:
        elapsed = max(1e-12, now - self._interval_start_time)
        effective_global_batch = (
            self._last_local_batch_size
            * self._last_world_size
            * self._last_accumulate_grad_batches
        )
        estimated_epoch_seconds = None
        if self.estimate_num_samples:
            estimated_epoch_seconds = self.estimate_num_samples / (
                self._interval_samples / elapsed
            )

        return {
            "global_step": int(trainer.global_step),
            "total_batches": int(self._total_batches),
            "world_size": self._last_world_size,
            "per_gpu_batch_size": self._last_local_batch_size,
            "accumulate_grad_batches": self._last_accumulate_grad_batches,
            "effective_global_batch_size": effective_global_batch,
            "interval_batches": int(self._interval_batches),
            "interval_samples": int(self._interval_samples),
            "samples_per_sec": self._interval_samples / elapsed,
            "steps_per_sec": self._interval_batches / elapsed,
            "data_loading_s_per_batch": self._current_data_time,
            "net_forward_s_per_batch": self._last_timing.get("net_forward_s", 0.0),
            "density_gradient_autograd_s_per_batch": self._last_timing.get(
                "density_gradient_autograd_s", 0.0
            ),
            "force_autograd_s_per_batch": self._last_timing.get("force_autograd_s", 0.0),
            "hvp_autograd_s_per_batch": self._last_timing.get("hvp_autograd_s", 0.0),
            "backward_s_per_batch": self._current_backward_time,
            "optimizer_s_per_batch": self._current_optimizer_time,
            "peak_gpu_memory_mb": self._max_cuda_memory_mb(),
            "estimated_epoch_seconds": estimated_epoch_seconds,
        }

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._interval_batches > 0:
            row = self._build_interval_row(trainer=trainer, now=self._now())
            self.rows.append(row)
            if trainer.is_global_zero:
                self._append_csv_row(row)
            self._interval_samples = 0
            self._interval_batches = 0

        if not trainer.is_global_zero:
            return
        wall_time_s = max(1e-12, self._now() - self._fit_start_time)
        measured_rows = self.rows
        mean_samples_per_sec = (
            sum(row["samples_per_sec"] for row in measured_rows) / len(measured_rows)
            if measured_rows
            else None
        )
        world_size = self._world_size(trainer)
        target_samples_per_sec = None
        estimates = {}
        if mean_samples_per_sec is not None:
            target_samples_per_sec = mean_samples_per_sec * self.target_num_devices / world_size
            if self.estimate_num_samples:
                for epochs in self.estimate_epochs:
                    seconds = self.estimate_num_samples * epochs / target_samples_per_sec
                    estimates[str(epochs)] = {
                        "seconds": seconds,
                        "hours": seconds / 3600.0,
                        "gpu_hours": seconds * self.target_num_devices / 3600.0,
                    }

        summary = {
            "world_size": world_size,
            "target_num_devices": self.target_num_devices,
            "total_batches": self._total_batches,
            "total_samples_estimated_global": self._total_samples,
            "wall_time_s": wall_time_s,
            "mean_samples_per_sec": mean_samples_per_sec,
            "target_samples_per_sec_linear_scaled": target_samples_per_sec,
            "estimate_num_samples": self.estimate_num_samples,
            "epoch_estimates": estimates,
            "peak_gpu_memory_mb": self._max_cuda_memory_mb(),
            "csv_path": str(self.csv_path) if self.csv_path is not None else None,
        }
        if self.json_path is not None:
            with self.json_path.open("w") as f:
                json.dump(summary, f, indent=2)
