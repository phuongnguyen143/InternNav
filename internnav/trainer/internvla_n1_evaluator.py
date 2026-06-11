'''
  Evaluator adapt from the trainer
  TODO:
  add more metrics
'''

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
torch.backends.cudnn.enabled = False
import transformers
from torch.utils.data import DataLoader
from torchvision.transforms import v2

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import internnav.dataset.internvla_n1_lerobot_dataset as lerobot_dataset
from internnav.dataset.internvla_n1_lerobot_dataset import make_supervised_data_module
from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    InternVLAN1ForCausalLM,
    InternVLAN1ModelConfig,
)
from internnav.trainer.internvla_n1_argument import (
    DataArguments,
    EvalArguments,
    ModelArguments,
)
from internnav.model.utils.tensorboard_utils import (
    TensorboardWriter,
    log_scalars_to_tensorboard,
    log_system_metrics_to_tensorboard,
)
from internnav.trainer.jetson_monitor import (
    format_memory_line,
    get_gpu_memory_mb,
    get_jetson_info,
    get_system_memory_mb,
    print_jetson_summary,
    print_status_block,
)


class RunningStats:
    """
    mean/std/min/max eval loss"""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0  # sum of squares of differences from the current mean
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    def variance(self) -> float:
        if self.n <= 1:
            return 0.0
        return self.m2 / (self.n - 1)

    def std(self) -> float:
        return self.variance() ** 0.5


def is_rank0() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _needs_deferred_system1_load(model_path: str, system1: str) -> bool:
    if not system1 or system1 == "none":
        return False
    return "internvla-n1-system2" in model_path.lower()


def load_model(model_args: ModelArguments, eval_args: EvalArguments):
    model_path = model_args.model_name_or_path
    dtype = torch.bfloat16 if eval_args.bf16 else None

    internvla_config = InternVLAN1ModelConfig.from_pretrained(
        model_path,
        cache_dir=eval_args.cache_dir,
    )
    if _needs_deferred_system1_load(model_path, model_args.system1):
        if is_rank0():
            print(
                f"Deferring System1 init during load "
                f"(checkpoint system1={internvla_config.system1!r}, "
                f"eval system1={model_args.system1!r})"
            )
        internvla_config.system1 = "none"

    model = InternVLAN1ForCausalLM.from_pretrained(
        model_path,
        config=internvla_config,
        cache_dir=eval_args.cache_dir,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if model_args.system1 and model_args.system1 != "none":
        inner = model.get_model()
        has_system1 = getattr(inner, "navdp", None) is not None or getattr(inner, "traj_dit", None) is not None
        if not has_system1:
            inner.initialize_vision_modules(model_args=model_args)

    model.eval()
    model.config.use_cache = False
    return model


def build_data_module(model_path: str, model_args: ModelArguments, data_args: DataArguments, eval_args: EvalArguments):
    if data_args.data_augmentation:
        data_args.transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ColorJitter(brightness=0.2, saturation=0.2),
                v2.RandomPosterize(bits=4),
                v2.RandomAdjustSharpness(sharpness_factor=1.5),
                v2.RandomAutocontrast(),
                v2.ToPILImage(),
                v2.Resize((data_args.resize_h, data_args.resize_w)),
            ]
        )
    else:
        data_args.transform_train = v2.Resize((data_args.resize_h, data_args.resize_w))

    data_args.image_processor = transformers.AutoProcessor.from_pretrained(model_path).image_processor
    data_args.model_type = "internvla-n1"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=eval_args.cache_dir,
        model_max_length=eval_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    return make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.dtype == torch.float64:
                value = value.to(torch.float32)
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _tensorboard_enabled(eval_args: EvalArguments) -> bool:
    report_to = os.environ.get("EVAL_REPORT_TO", eval_args.report_to)
    return "tensorboard" in {x.strip().lower() for x in report_to.split(",") if x.strip()}


def _tensorboard_log_dir(output_dir: str, eval_args: EvalArguments) -> str:
    if eval_args.logging_dir:
        return eval_args.logging_dir
    env_dir = os.environ.get("EVAL_TENSORBOARD_DIR", "").strip()
    if env_dir:
        return env_dir
    return os.path.join(output_dir, "tensorboard")


def create_eval_tensorboard_writer(
    output_dir: str,
    eval_args: EvalArguments,
) -> Optional[TensorboardWriter]:
    if not is_rank0() or not _tensorboard_enabled(eval_args):
        return None
    log_dir = _tensorboard_log_dir(output_dir, eval_args)
    os.makedirs(log_dir, exist_ok=True)
    print(f"tensorboard log_dir: {log_dir}")
    return TensorboardWriter(log_dir)


def my_eval(
    model: InternVLAN1ForCausalLM,
    dataloader: DataLoader,
    eval_args: EvalArguments,
    output_dir: str,
) -> Dict[str, float]:
    device = torch.device("cuda")
    model.to(device)

    log_name = os.environ.get("EVAL_LOG_FILE", "eval_metrics.jsonl")
    log_path = os.path.join(output_dir, log_name)
    print(f"log_path: {log_path}")
    if is_rank0() and os.path.exists(log_path):
        os.remove(log_path)

    tb_writer = create_eval_tensorboard_writer(output_dir, eval_args)

    total_loss = 0.0
    total_batches = 0
    started = time.time()

    loss_alert_multiplier = float(os.environ.get("LOSS_ALERT_MULTIPLIER", "5"))
    loss_baseline_steps = int(os.environ.get("LOSS_BASELINE_STEPS", "5"))


    stats = RunningStats()
    baseline_mean: Optional[float] = None
    spikes = 0
    spikes_by_ratio = 0

    if is_rank0():
        max_steps = eval_args.max_eval_steps if eval_args.max_eval_steps > 0 else len(dataloader)
        print(f"[eval] running up to {max_steps} batch(es) on {device}")
        begin_record = {
            "event": "eval_begin",
            "timestamp": time.time(),
            "output_dir": output_dir,
            "max_eval_steps": eval_args.max_eval_steps,
            "dataset_batches": len(dataloader),
            "per_device_eval_batch_size": eval_args.per_device_eval_batch_size,
            "bf16": eval_args.bf16,
            "loss_alert_multiplier": loss_alert_multiplier,
            "loss_baseline_steps": loss_baseline_steps,
        }
        append_jsonl(log_path, begin_record)
        if tb_writer is not None:
            tb_writer.add_text("eval/config", json.dumps(begin_record, default=str, indent=2), 0)
            log_scalars_to_tensorboard(
                tb_writer,
                0,
                {
                    "max_eval_steps": eval_args.max_eval_steps,
                    "dataset_batches": len(dataloader),
                    "per_device_eval_batch_size": eval_args.per_device_eval_batch_size,
                },
            )

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if eval_args.max_eval_steps > 0 and step >= eval_args.max_eval_steps:
                break

            batch = move_batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=eval_args.bf16 and device.type == "cuda",
            ):
                outputs = model(**batch)

            loss = outputs.loss
            if loss is None:
                raise RuntimeError(
                    "Model returned loss=None — ensure pixel_goal_only=True and system1 is set for dual-system eval."
                )

            loss_value = float(loss.item())
            total_loss += loss_value
            total_batches += 1
            stats.update(loss_value)

            if baseline_mean is None and stats.n >= loss_baseline_steps:
                baseline_mean = stats.mean
                if is_rank0():
                    baseline_record = {
                        "event": "loss_baseline_set",
                        "timestamp": time.time(),
                        "baseline_steps": loss_baseline_steps,
                        "baseline_mean": baseline_mean,
                        "loss_alert_multiplier": loss_alert_multiplier,
                    }
                    append_jsonl(log_path, baseline_record)
                    log_scalars_to_tensorboard(
                        tb_writer,
                        total_batches,
                        {
                            "baseline_mean": baseline_mean,
                            "loss_alert_multiplier": loss_alert_multiplier,
                        },
                        prefix="eval/baseline",
                    )

            if is_rank0():
                running_avg = stats.mean
                running_std = stats.std()
                ratio_to_avg = loss_value / (running_avg + 1e-12)
                ratio_to_baseline = (
                    (loss_value / (baseline_mean + 1e-12)) if baseline_mean is not None else None
                )

                alert = False
                alert_reason: Optional[str] = None
                if baseline_mean is not None and loss_value > baseline_mean * loss_alert_multiplier:
                    alert = True
                    alert_reason = "ratio"
                    spikes_by_ratio += 1

                if alert:
                    spikes += 1

                print_prefix = "**LOSS ALERT** " if alert else "[eval]"
                print(
                    f"{print_prefix} step {total_batches} loss={loss_value:.4f} "
                    f"avg={running_avg:.4f} std={running_std:.4f} "
                    f"min={stats.min:.4f} max={stats.max:.4f} "
                    f"ratio_to_avg={ratio_to_avg:.2f} "
                    f"ratio_to_baseline={ratio_to_baseline:.2f}" if ratio_to_baseline is not None else
                    f"{print_prefix} step {total_batches} loss={loss_value:.4f} "
                    f"avg={running_avg:.4f} std={running_std:.4f} "
                    f"min={stats.min:.4f} max={stats.max:.4f} "
                    f"ratio_to_avg={ratio_to_avg:.2f}"
                )
                step_record = {
                    "event": "eval_step",
                    "timestamp": time.time(),
                    "step": total_batches,
                    "loss": loss_value,
                    "avg_loss": running_avg,
                    "std_loss": running_std,
                    "min_loss_so_far": stats.min,
                    "max_loss_so_far": stats.max,
                    "ratio_to_running_avg": ratio_to_avg,
                    "baseline_mean": baseline_mean,
                    "ratio_to_baseline": ratio_to_baseline,
                    "alert": alert,
                    "alert_reason": alert_reason,
                    "loss_spikes_total": spikes,
                    "loss_spikes_by_ratio": spikes_by_ratio,
                    "elapsed_sec": time.time() - started,
                }
                append_jsonl(log_path, step_record)
                log_scalars_to_tensorboard(
                    tb_writer,
                    total_batches,
                    {
                        "loss": loss_value,
                        "avg_loss": running_avg,
                        "std_loss": running_std,
                        "min_loss_so_far": stats.min,
                        "max_loss_so_far": stats.max,
                        "ratio_to_running_avg": ratio_to_avg,
                        "baseline_mean": baseline_mean,
                        "ratio_to_baseline": ratio_to_baseline,
                        "alert": alert,
                        "loss_spikes_total": spikes,
                        "loss_spikes_by_ratio": spikes_by_ratio,
                        "elapsed_sec": step_record["elapsed_sec"],
                    },
                )
                if total_batches == 1 or total_batches % 10 == 0 or alert:
                    log_system_metrics_to_tensorboard(
                        tb_writer,
                        total_batches,
                        get_system_memory_mb(),
                        get_gpu_memory_mb(),
                        get_jetson_info(),
                    )
                    print_status_block("eval", step=total_batches, loss=running_avg)

    if total_batches == 0:
        raise RuntimeError("eval dataloader produced 0 batches — check dataset paths and VLN_DATASETS.")

    metrics = {
        "eval_loss": total_loss / total_batches,
        "eval_batches": float(total_batches),
        "eval_runtime_sec": time.time() - started,
        "eval_loss_std": stats.std(),
        "eval_loss_min": stats.min,
        "eval_loss_max": stats.max,
        "loss_spikes_total": float(spikes),
        "loss_spikes_by_ratio": float(spikes_by_ratio),
        "loss_baseline_mean": baseline_mean if baseline_mean is not None else float("nan"),
    }

    if is_rank0():
        end_record = {
            "event": "eval_end",
            "timestamp": time.time(),
            **metrics,
        }
        append_jsonl(log_path, end_record)
        final_step = int(metrics["eval_batches"])
        log_scalars_to_tensorboard(tb_writer, final_step, metrics, prefix="eval/summary")
        log_system_metrics_to_tensorboard(
            tb_writer,
            final_step,
            get_system_memory_mb(),
            get_gpu_memory_mb(),
            get_jetson_info(),
        )
        if tb_writer is not None:
            tb_writer.add_text("eval/summary", json.dumps(end_record, default=str, indent=2), final_step)
            tb_writer.close()
        print("\n=== Evaluation finished ===")
        print(f"  batches:   {int(metrics['eval_batches'])}")
        print(f"  eval_loss: {metrics['eval_loss']:.6f}")
        print(f"  min/max:   {metrics['eval_loss_min']:.6f} / {metrics['eval_loss_max']:.6f}")
        print(f"  std:       {metrics['eval_loss_std']:.6f}")
        print(f"  spikes:    {int(metrics['loss_spikes_total'])} (ratio={int(metrics['loss_spikes_by_ratio'])})")
        print(f"  runtime:   {metrics['eval_runtime_sec']:.1f}s")
        print(f"  log_file:  {log_path}")
        if tb_writer is not None and tb_writer.writer is not None:
            print(f"  tensorboard: {_tensorboard_log_dir(output_dir, eval_args)}")
        print(format_memory_line("mem"))
        print("=== End evaluation ===\n")

    return metrics


def evaluate(attn_implementation: str = "sdpa"):
    del attn_implementation  # kept for symmetry with trainer entry point

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    os.makedirs(eval_args.output_dir, exist_ok=True)
    lerobot_dataset.local_rank = eval_args.local_rank

    if is_rank0():
        print_jetson_summary()
        print("\n=== Evaluation data env ===")
        print(f"model_name_or_path: {model_args.model_name_or_path}")
        print(f"vln_dataset_use: {data_args.vln_dataset_use}")
        print(f"system1: {model_args.system1}")
        print(f"INTERNAV_R2R_DATA_PATH: {os.environ.get('INTERNAV_R2R_DATA_PATH', '(not set)')}")
        print(f"INTERNAV_RXR_DATA_PATH: {os.environ.get('INTERNAV_RXR_DATA_PATH', '(not set)')}")
        print(f"INTERNAV_SCALEVLN_DATA_PATH: {os.environ.get('INTERNAV_SCALEVLN_DATA_PATH', '(not set)')}")
        print("=== End evaluation data env ===\n")

    model = load_model(model_args, eval_args)
    data_module = build_data_module(model_args.model_name_or_path, model_args, data_args, eval_args)
    dataset = data_module["train_dataset"]
    collator = data_module["data_collator"]

    if is_rank0():
        print(f"eval_dataset size: {len(dataset)}")
        print_status_block("after_dataset_load")

    if len(dataset) == 0:
        raise RuntimeError("eval dataset has 0 samples — check dataset debug output and data paths.")

    dataloader = DataLoader(
        dataset,
        batch_size=eval_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=eval_args.dataloader_num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    my_eval(model, dataloader, eval_args, eval_args.output_dir)


if __name__ == "__main__":
    evaluate()
