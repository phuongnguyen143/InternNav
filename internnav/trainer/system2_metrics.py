"""Training-time metrics for System 2 (Qwen2.5-VL) navigation supervision."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import torch

from internnav.dataset.internvla_n1_lerobot_dataset import IGNORE_INDEX, TRAJ_TOKEN_INDEX


def _normalize_action_text(text: str) -> str:
    """Collapse whitespace and keep only navigation action symbols."""
    text = text.strip().replace(" ", "").replace("\n", "")
    if not text:
        return text
    if "STOP" in text.upper():
        return "STOP"
    out = []
    for ch in text:
        if ch in "↑←→↓":
            out.append(ch)
    return "".join(out)


def _decode_supervised_tokens(
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch_idx: int,
    tokenizer: Any,
) -> tuple[str, str]:
    """Decode predicted vs ground-truth text on supervised (non -100) label positions."""
    shift_logits = logits[batch_idx, :-1]
    shift_labels = labels[batch_idx, 1:]
    # Traj latent slots are appended to labels but are not decodable text tokens.
    mask = (shift_labels != IGNORE_INDEX) & (shift_labels != TRAJ_TOKEN_INDEX)
    if not mask.any():
        return "", ""

    pred_ids = shift_logits.argmax(dim=-1)[mask]
    label_ids = shift_labels[mask]
    pred_text = tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)
    label_text = tokenizer.decode(label_ids.tolist(), skip_special_tokens=True)
    return pred_text, label_text


def _parse_pixel_coords(text: str) -> Optional[tuple[float, float]]:
    """Extract the first two integers from decoded text as (x, y)."""
    nums = re.findall(r"\d+", text)
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


def compute_system2_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_types: Sequence[str],
    pixel_coords_gt: Optional[torch.Tensor],
    tokenizer: Any,
) -> Dict[str, float]:
    """Compute turn accuracy and pixel coordinate L2 error for one batch.

    Returns scalar dict suitable for Trainer.log() / TensorBoard.
    """
    batch_size = labels.shape[0]
    turn_correct = 0
    turn_total = 0
    stop_correct = 0
    stop_total = 0
    coord_l2_sum = 0.0
    coord_count = 0
    coord_parse_ok = 0
    pixel_total = 0

    for b in range(batch_size):
        sample_type = sample_types[b]
        pred_text, label_text = _decode_supervised_tokens(logits, labels, b, tokenizer)

        # print("pixel_coords_gt", pixel_coords_gt, "\n")

        # print("sample_type", sample_type, "\n")
        # print("------------START--------------------")
        # print("pred_text: ",  _normalize_action_text(pred_text))
        # print("--------------------------------")
        # print("label_text: ",  _normalize_action_text(label_text))
        # print("same: ",  _normalize_action_text(pred_text) == _normalize_action_text(label_text))
        # print("--------------------------------")

        if sample_type == "turn":
            turn_total += 1
            if _normalize_action_text(pred_text) == _normalize_action_text(label_text):
                turn_correct += 1
        elif sample_type == "stop":
            stop_total += 1
            if _normalize_action_text(pred_text) == _normalize_action_text(label_text):
                stop_correct += 1
        elif sample_type == "pixel_goal":
            pixel_total += 1
            if pixel_coords_gt is None:
                continue
            gt_x, gt_y = pixel_coords_gt[b].tolist()
            if not (gt_x == gt_x and gt_y == gt_y):  # skip NaN rows
                continue
            pred_coords = _parse_pixel_coords(pred_text) # pred_coords:  (290.0, 178.0)
            if pred_coords is None:
                continue
            coord_parse_ok += 1
            px, py = pred_coords
            coord_l2_sum += ((px - gt_x) ** 2 + (py - gt_y) ** 2) ** 0.5
            coord_count += 1

    metrics: Dict[str, float] = {}

    if turn_total > 0:
        metrics["turn_accuracy"] = turn_correct / turn_total
    if stop_total > 0:
        metrics["stop_accuracy"] = stop_correct / stop_total
    if turn_total + stop_total > 0:
        metrics["discrete_action_accuracy"] = (turn_correct + stop_correct) / (turn_total + stop_total)

    if pixel_total > 0:
        metrics["pixel_coord_parse_rate"] = coord_parse_ok / pixel_total
    if coord_count > 0:
        metrics["pixel_coord_l2"] = coord_l2_sum / coord_count

    return metrics
