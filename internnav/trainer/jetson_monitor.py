"""Memory, Jetson performance, and training metrics logging for edge training."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from typing import Any, Dict, Optional

import torch

from internnav.model.utils.tensorboard_utils import (
    TensorboardWriter,
    log_scalars_to_tensorboard,
    log_system_metrics_to_tensorboard,
)

TRAIN_METRIC_KEYS = (
    "loss",
    "train_loss",
    "learning_rate",
    "epoch",
    "grad_norm",
    "eval_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
)


def _read_kb(path: str, key: str) -> Optional[int]:
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def get_system_memory_mb() -> Dict[str, float]:
    total_kb = _read_kb("/proc/meminfo", "MemTotal:")
    avail_kb = _read_kb("/proc/meminfo", "MemAvailable:")
    if total_kb is None:
        return {}
    used_kb = total_kb - (avail_kb or 0)
    return {
        "total_mb": total_kb / 1024,
        "used_mb": used_kb / 1024,
        "avail_mb": (avail_kb or 0) / 1024,
        "used_pct": 100.0 * used_kb / total_kb,
    }


def get_gpu_memory_mb(device: int = 0) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    try:
        torch.cuda.synchronize(device)
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        return {
            "alloc_mb": alloc / (1024**2),
            "reserved_mb": reserved / (1024**2),
            "total_mb": total / (1024**2),
            "free_mb": (total - reserved) / (1024**2),
            "used_pct": 100.0 * reserved / total if total else 0.0,
        }
    except Exception:
        return {}


def _run_cmd(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=3)
        return out.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_jetson_info() -> Dict[str, str]:
    info: Dict[str, str] = {"hostname": platform.node()}

    tegra = _run_cmd(["cat", "/etc/nv_tegra_release"])
    if tegra:
        info["tegra_release"] = tegra.splitlines()[0]

    jetson_release = _run_cmd(["jetson_release"])
    if jetson_release:
        info["jetson_release"] = jetson_release.splitlines()[0]

    nvpmodel = _run_cmd(["nvpmodel", "-q"])
    if nvpmodel:
        for line in nvpmodel.splitlines():
            if "NV Power Mode" in line or "power mode" in line.lower():
                info["power_mode"] = line.strip()
                break

    for i, path in enumerate(
        [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone1/temp",
        ]
    ):
        try:
            with open(path, "r") as f:
                temp_c = int(f.read().strip()) / 1000.0
            info[f"temp_zone{i}_c"] = f"{temp_c:.1f}"
        except OSError:
            pass

    smi = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if smi:
        parts = [p.strip() for p in smi.split(",")]
        if len(parts) >= 6:
            info["gpu_name"] = parts[0]
            info["gpu_driver"] = parts[1]
            info["gpu_util_pct"] = parts[2]
            info["gpu_mem_util_pct"] = parts[3]
            info["gpu_temp_c"] = parts[4]
            info["gpu_power_w"] = parts[5]

    return info


def _fmt_float(value: Any, precision: int = 4) -> str:
    try:
        return f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return str(value)


def extract_training_metrics(logs: Optional[Dict[str, Any]], state) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "global_step": state.global_step,
        "epoch": round(float(state.epoch), 6) if state.epoch is not None else None,
        "max_steps": state.max_steps,
    }
    if logs:
        for key in TRAIN_METRIC_KEYS:
            if key in logs:
                metrics[key] = logs[key]
        if "loss" not in metrics and "train_loss" in metrics:
            metrics["loss"] = metrics["train_loss"]
    if hasattr(state, "num_input_tokens_seen") and state.num_input_tokens_seen:
        metrics["tokens_seen"] = state.num_input_tokens_seen
    return metrics


def format_training_line(metrics: Dict[str, Any]) -> str:
    parts = ["[train]"]
    if metrics.get("global_step") is not None:
        parts.append(f"step={metrics['global_step']}")
    if metrics.get("max_steps"):
        parts.append(f"/{metrics['max_steps']}")
    if metrics.get("epoch") is not None:
        parts.append(f"epoch={_fmt_float(metrics['epoch'], 4)}")
    if metrics.get("loss") is not None:
        parts.append(f"loss={_fmt_float(metrics['loss'], 4)}")
    if metrics.get("learning_rate") is not None:
        parts.append(f"lr={_fmt_float(metrics['learning_rate'], 6)}")
    if metrics.get("grad_norm") is not None:
        parts.append(f"grad_norm={_fmt_float(metrics['grad_norm'], 4)}")
    if metrics.get("train_steps_per_second") is not None:
        parts.append(f"steps/s={_fmt_float(metrics['train_steps_per_second'], 3)}")
    if metrics.get("train_samples_per_second") is not None:
        parts.append(f"samples/s={_fmt_float(metrics['train_samples_per_second'], 3)}")
    if metrics.get("tokens_seen") is not None:
        parts.append(f"tokens={metrics['tokens_seen']}")
    return " | ".join(parts)


def format_memory_line(tag: str = "mem") -> str:
    sys_mem = get_system_memory_mb()
    gpu_mem = get_gpu_memory_mb()
    jetson = get_jetson_info()

    parts = [f"[{tag}]"]
    if sys_mem:
        parts.append(
            f"RAM {sys_mem['used_mb']:.0f}/{sys_mem['total_mb']:.0f}MB"
            f" ({sys_mem['used_pct']:.1f}%)"
        )
    if gpu_mem:
        parts.append(
            f"GPU {gpu_mem['reserved_mb']:.0f}/{gpu_mem['total_mb']:.0f}MB"
            f" alloc={gpu_mem['alloc_mb']:.0f}MB ({gpu_mem['used_pct']:.1f}%)"
        )
    for key in ("gpu_util_pct", "gpu_mem_util_pct", "gpu_temp_c", "gpu_power_w", "temp_zone0_c"):
        if key in jetson:
            label = key.replace("_", " ")
            parts.append(f"{label}={jetson[key]}")
    return " | ".join(parts)


def print_status_block(title: str, step: Optional[int] = None, loss: Optional[float] = None) -> None:
    parts = [f"[{title}]"]
    if step is not None:
        parts.append(f"step={step}")
    if loss is not None:
        parts.append(f"loss={_fmt_float(loss, 4)}")
    print(" | ".join(parts), flush=True)
    print(format_memory_line("mem"), flush=True)


def print_jetson_summary() -> None:
    print("\n=== Jetson / system summary ===", flush=True)
    jetson = get_jetson_info()
    for key, value in jetson.items():
        print(f"  {key}: {value}")
    sys_mem = get_system_memory_mb()
    if sys_mem:
        print(
            f"  RAM: {sys_mem['used_mb']:.0f}/{sys_mem['total_mb']:.0f} MB used"
            f" ({sys_mem['avail_mb']:.0f} MB avail)"
        )
    gpu_mem = get_gpu_memory_mb()
    if gpu_mem:
        print(
            f"  CUDA: {gpu_mem['reserved_mb']:.0f}/{gpu_mem['total_mb']:.0f} MB reserved"
            f" ({gpu_mem['alloc_mb']:.0f} MB allocated)"
        )
    if torch.cuda.is_available():
        print(f"  torch: {torch.__version__}  cuda: {torch.version.cuda}")
        print(f"  device: {torch.cuda.get_device_name(0)}")
    print("=== End Jetson summary ===\n", flush=True)


def train_tensorboard_enabled(args) -> bool:
    """Custom TensorBoard (loss + Jetson metrics), independent of HF report_to."""
    env_flag = os.environ.get("TRAIN_TENSORBOARD", "").strip().lower()
    if env_flag in ("0", "false", "no", "off"):
        return False
    if env_flag in ("1", "true", "yes", "on"):
        return True
    report_to = os.environ.get("TRAIN_REPORT_TO", "").strip()
    if report_to:
        tokens = {x.strip().lower() for x in report_to.split(",") if x.strip()}
        return "tensorboard" in tokens or "all" in tokens
    return True


def train_tensorboard_log_dir(args) -> str:
    logging_dir = getattr(args, "logging_dir", None)
    if logging_dir:
        return logging_dir
    env_dir = os.environ.get("TRAIN_TENSORBOARD_DIR", "").strip()
    if env_dir:
        return env_dir
    return os.path.join(args.output_dir, "tensorboard")


class JetsonTrainingCallback:
    def __init__(self):
        self.log_path: Optional[str] = None
        self.tb_writer: Optional[TensorboardWriter] = None
        self._train_start: Optional[float] = None
        self._last_step_time: Optional[float] = None
        self._last_step: int = 0

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._train_start = time.time()
        self._last_step_time = self._train_start
        self._last_step = 0
        log_name = os.environ.get("TRAIN_LOG_FILE", "training_metrics.jsonl")
        self.log_path = os.path.join(args.output_dir, log_name)
        header = {
            "event": "train_begin",
            "timestamp": time.time(),
            "output_dir": args.output_dir,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "deepspeed": bool(args.deepspeed),
            "bf16": args.bf16,
            "model_max_length": args.model_max_length,
        }
        self._append_jsonl(header)
        if train_tensorboard_enabled(args):
            log_dir = train_tensorboard_log_dir(args)
            os.makedirs(log_dir, exist_ok=True)
            self.tb_writer = TensorboardWriter(log_dir)
            print(f"  tensorboard: {log_dir}", flush=True)
        print("\n=== Training started ===", flush=True)
        print(
            f"  log_file: {self.log_path}\n"
            f"  lr={args.learning_rate} batch={args.per_device_train_batch_size} "
            f"grad_accum={args.gradient_accumulation_steps} "
            f"epochs={args.num_train_epochs} max_steps={args.max_steps}",
            flush=True,
        )
        print(format_memory_line("mem"), flush=True)
        print("=== End training start ===\n", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        now = time.time()
        metrics = extract_training_metrics(logs, state)
        if self._last_step_time and state.global_step > self._last_step:
            step_delta = state.global_step - self._last_step
            time_delta = now - self._last_step_time
            if step_delta > 0 and time_delta > 0:
                metrics["steps_per_sec_recent"] = step_delta / time_delta
        self._last_step = state.global_step
        self._last_step_time = now
        if self._train_start:
            metrics["elapsed_sec"] = now - self._train_start

        record = {
            "event": "log",
            "timestamp": now,
            **metrics,
            "system_memory": get_system_memory_mb(),
            "gpu_memory": get_gpu_memory_mb(),
            "jetson": get_jetson_info(),
        }
        self._append_jsonl(record)
        print(format_training_line(metrics), flush=True)
        print(format_memory_line("mem"), flush=True)
        if self.tb_writer is not None:
            step = state.global_step
            log_scalars_to_tensorboard(self.tb_writer, step, metrics, prefix="train")
            if logs:
                extra = {
                    k: v
                    for k, v in logs.items()
                    if k not in metrics and not isinstance(v, (dict, list, tuple, str))
                }
                log_scalars_to_tensorboard(self.tb_writer, step, extra, prefix="train")
            log_system_metrics_to_tensorboard(
                self.tb_writer,
                step,
                record["system_memory"],
                record["gpu_memory"],
                record["jetson"],
            )

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        record = {
            "event": "checkpoint_saved",
            "timestamp": time.time(),
            "global_step": state.global_step,
            "checkpoint_dir": ckpt_dir,
        }
        self._append_jsonl(record)
        print(f"[checkpoint] step={state.global_step} saved to {ckpt_dir}", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        record = {
            "event": "train_end",
            "timestamp": time.time(),
            "global_step": state.global_step,
            "epoch": state.epoch,
            "elapsed_sec": time.time() - self._train_start if self._train_start else None,
            "system_memory": get_system_memory_mb(),
            "gpu_memory": get_gpu_memory_mb(),
        }
        self._append_jsonl(record)
        print("\n=== Training finished ===", flush=True)
        print(
            f"  steps={state.global_step} epoch={_fmt_float(state.epoch, 4)} "
            f"elapsed={record['elapsed_sec']:.1f}s",
            flush=True,
        )
        print(f"  metrics log: {self.log_path}", flush=True)
        if self.tb_writer is not None:
            print(f"  tensorboard: {train_tensorboard_log_dir(args)}", flush=True)
            if self.tb_writer.writer:
                self.tb_writer.writer.close()
            self.tb_writer = None
        print(format_memory_line("mem"), flush=True)
        print("=== End training ===\n", flush=True)
