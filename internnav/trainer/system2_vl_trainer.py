"""HuggingFace Trainer with System 2 navigation metrics logged to TensorBoard."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch
from transformers import Trainer
from transformers.modeling_outputs import ModelOutput

from internnav.trainer.system2_metrics import compute_system2_metrics


class System2VLTrainer(Trainer):
    """Extends HF Trainer to log turn accuracy and pixel coordinate L2 during training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_s2_metrics: list[Dict[str, float]] = []

    def _should_compute_s2_metrics(self) -> bool:
        every = int(os.environ.get("S2_METRICS_STEPS", "1"))
        if every <= 0:
            return False
        if hasattr(self, "accelerator") and getattr(self.accelerator, "sync_gradients", True) is False:
            return False
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps != 0:
            return False
        if every > 1 and self.state.global_step % every != 0:
            return False
        return True

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if self._pending_s2_metrics:
            merged: Dict[str, float] = {}
            for key in self._pending_s2_metrics[0]:
                values = [m[key] for m in self._pending_s2_metrics if key in m]
                if values:
                    merged[key] = sum(values) / len(values)
            logs.update(merged)
            self._pending_s2_metrics.clear()
        super().log(logs, start_time=start_time)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ModelOutput]]:
        sample_types = inputs.pop("sample_types", None)
        pixel_coords_gt = inputs.get("pixel_coords_gt")

        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, ModelOutput) else outputs[0]

        if (
            sample_types is not None
            and self.model.training
            and hasattr(outputs, "logits")
            and outputs.logits is not None
            and self._should_compute_s2_metrics()
        ):
            tokenizer = self.processing_class
            if tokenizer is not None:
                with torch.no_grad():
                    metrics = compute_system2_metrics(
                        outputs.logits,
                        inputs["labels"],
                        sample_types,
                        pixel_coords_gt,
                        tokenizer,
                    )
                if metrics:
                    self._pending_s2_metrics.append(metrics)

        return (loss, outputs) if return_outputs else loss
