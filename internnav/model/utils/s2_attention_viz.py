
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image


@dataclass
class S2AttentionBundle:

    query_positions: list[int]
    layer_indices: list[int]
    # layer_idx -> list of (h, w) heatmaps, one per input image
    vision_heatmaps_per_layer: dict[int, list[np.ndarray]]
    text_token_mask: np.ndarray
    vision_token_mask: np.ndarray
    input_token_strings: list[str]
    llm_output: str
    instruction: str = ""
    instruction_token_mask: np.ndarray | None = None
    instruction_mass_per_layer: dict[int, float] | None = None
    prompt_text_positions: list[int] | None = None
    prompt_text_string: str = ""
    instruction_char_span: tuple[int, int] | None = None
    # layer_idx -> attention weights on each prompt text token
    prompt_text_attn_per_layer: dict[int, np.ndarray] | None = None


@dataclass
class SceneInstructionAttentionStep:
    frame_index: int
    layer_index: int
    instruction_mass: float


class SceneInstructionAttentionTracker:
    """Collect instruction attention across S2 re-plans in one scene run."""

    def __init__(self, layer_index: int | None = None):
        self.layer_index = layer_index
        self.token_steps: list[tuple[int, int, np.ndarray]] = []
        self.mass_steps: list[SceneInstructionAttentionStep] = []

    def add(self, frame_index: int, bundle: S2AttentionBundle) -> None:
        layer = (
            self.layer_index
            if self.layer_index is not None
            else bundle.layer_indices[-1]
        )

        if bundle.instruction_mass_per_layer:
            self.mass_steps.append(
                SceneInstructionAttentionStep(
                    frame_index=frame_index,
                    layer_index=layer,
                    instruction_mass=float(bundle.instruction_mass_per_layer.get(layer, 0.0)),
                )
            )

        if (
            bundle.prompt_text_attn_per_layer
            and bundle.prompt_text_positions
            and bundle.instruction_token_mask is not None
        ):
            instr_attn = bundle.prompt_text_attn_per_layer.get(layer)
            if instr_attn is not None:
                pos_to_idx = {pos: idx for idx, pos in enumerate(bundle.prompt_text_positions)}
                instr_indices = [
                    pos_to_idx[pos]
                    for pos in bundle.prompt_text_positions
                    if bundle.instruction_token_mask[pos]
                ]
                if instr_indices:
                    self.token_steps.append(
                        (frame_index, layer, instr_attn[instr_indices].copy())
                    )

    def save(self, output_dir: str) -> dict[str, str]:
        import os

        os.makedirs(output_dir, exist_ok=True)
        saved: dict[str, str] = {}
        if self.token_steps:
            saved.update(self._save_token_scene_heatmap(output_dir))
        if self.mass_steps:
            saved.update(self._save_mass_over_time(output_dir))
        return saved

    def _save_token_scene_heatmap(self, output_dir: str) -> dict[str, str]:
        import os

        frame_labels = [f"frame {frame_idx}" for frame_idx, _, _ in self.token_steps]
        matrix = np.stack([weights for _, _, weights in self.token_steps], axis=0)
        row_max = matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        matrix_norm = matrix / row_max

        layer = self.token_steps[0][1]
        fig_h = max(4, 0.45 * len(self.token_steps))
        fig_w = max(10, 0.08 * matrix.shape[1])
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(matrix_norm, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(frame_labels)))
        ax.set_yticklabels(frame_labels)
        ax.set_xlabel("Instruction tokens")
        ax.set_ylabel("S2 re-plan step")
        ax.set_title(
            f"Instruction token attention across scene (layer {layer}, rows normalized)"
        )
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "scene_instruction_token_attention_heatmap.png")
        Image.fromarray(_figure_to_rgb(fig)).save(path)
        return {"scene_instruction_token_heatmap": path}

    def _save_mass_over_time(self, output_dir: str) -> dict[str, str]:
        import os

        layer = self.mass_steps[0].layer_index
        masses = [s.instruction_mass for s in self.mass_steps]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([s.frame_index for s in self.mass_steps], masses, marker="o")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Instruction attention mass")
        ax.set_title(f"Total instruction attention over scene (layer {layer})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        line_path = os.path.join(output_dir, "scene_instruction_attention_over_time.png")
        Image.fromarray(_figure_to_rgb(fig)).save(line_path)
        return {"scene_instruction_over_time": line_path}


def _resolve_layer_indices(num_layers: int, layers: Sequence[int] | None) -> list[int]:
    if not layers:
        return [0, num_layers // 2, num_layers - 1]
    resolved = []
    for layer in layers:
        idx = int(layer)
        if idx < 0:
            idx = num_layers + idx
        idx = max(0, min(num_layers - 1, idx))
        resolved.append(idx)
    return sorted(set(resolved))


def _build_token_masks(input_ids: torch.Tensor, image_token_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (seq_len,) for vision and text tokens (batch 0)."""
    ids = input_ids[0].detach().cpu().numpy()
    vision_mask = ids == image_token_id
    text_mask = ~vision_mask
    return text_mask, vision_mask


def _find_query_positions(
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    prompt_len: int,
    tokenizer: Any,
    llm_output: str,
) -> list[int]:
    """Pick query token positions for attention analysis."""
    seq = output_ids[0].detach().cpu().tolist()
    gen_ids = seq[prompt_len:]

    digit_positions = []
    for offset, token_id in enumerate(gen_ids):
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        if any(ch.isdigit() for ch in piece):
            digit_positions.append(prompt_len + offset)

    if digit_positions:
        return digit_positions

    stop_variants = {"stop", "STOP", "Stop"}
    if llm_output.strip() in stop_variants or "STOP" in llm_output:
        for offset, token_id in enumerate(gen_ids):
            piece = tokenizer.decode([token_id], skip_special_tokens=True)
            if "STOP" in piece.upper():
                return [prompt_len + offset]

    if gen_ids:
        return [prompt_len + len(gen_ids) - 1]

    return [max(prompt_len - 1, 0)]


def _prompt_text_positions(
    full_ids: torch.Tensor,
    text_mask: np.ndarray,
    prompt_len: int,
) -> list[int]:
    return [i for i in range(prompt_len) if text_mask[i]]


def _decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _token_offsets(tokenizer: Any, token_ids: list[int]) -> list[tuple[int, int]]:
    """Return character offsets for each token id in order."""
    text = _decode_ids(tokenizer, token_ids)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) == list(token_ids):
        return enc["offset_mapping"]

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for tok_id in token_ids:
        piece = tokenizer.decode([tok_id], skip_special_tokens=False)
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return offsets


def _find_instruction_char_span(prompt_text: str, instruction: str) -> tuple[int, int] | None:
    instruction = instruction.strip()
    if not instruction:
        return None

    for candidate in (instruction, instruction.rstrip(".")):
        start = prompt_text.find(candidate)
        if start >= 0:
            return start, start + len(candidate)

    lowered_prompt = prompt_text.lower()
    lowered_instr = instruction.lower().rstrip(".")
    start = lowered_prompt.find(lowered_instr)
    if start >= 0:
        return start, start + len(lowered_instr)

    marker = "your task is to "
    marker_idx = lowered_prompt.find(marker)
    if marker_idx >= 0:
        start = marker_idx + len(marker)
        end = start + len(instruction)
        return start, min(end, len(prompt_text))

    return None


def _span_to_token_mask(
    seq_len: int,
    text_positions: list[int],
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> np.ndarray:
    mask = np.zeros(seq_len, dtype=bool)
    for pos, (tok_start, tok_end) in zip(text_positions, offsets):
        if tok_end > char_start and tok_start < char_end:
            mask[pos] = True
    return mask


def _find_instruction_token_mask(
    full_ids: torch.Tensor,
    text_mask: np.ndarray,
    instruction: str,
    tokenizer: Any,
    prompt_len: int,
) -> tuple[np.ndarray, list[int], str, tuple[int, int] | None]:
    """Locate instruction tokens using decoded prompt text + character offsets."""
    seq_len = full_ids.shape[1]
    instr_mask = np.zeros(seq_len, dtype=bool)
    ids = full_ids[0].detach().cpu().tolist()
    text_positions = _prompt_text_positions(full_ids, text_mask, prompt_len)
    if not text_positions:
        return instr_mask, text_positions, "", None

    text_ids = [ids[pos] for pos in text_positions]
    prompt_text = _decode_ids(tokenizer, text_ids)
    offsets = _token_offsets(tokenizer, text_ids)
    char_span = _find_instruction_char_span(prompt_text, instruction)
    if char_span is not None:
        instr_mask = _span_to_token_mask(seq_len, text_positions, offsets, *char_span)
        if instr_mask.sum() == 0:
            # Offset alignment can fail when decode/encode round-trips differ; use proportional fallback.
            start_frac = char_span[0] / max(len(prompt_text), 1)
            end_frac = char_span[1] / max(len(prompt_text), 1)
            n_text = len(text_positions)
            t0 = int(start_frac * n_text)
            t1 = max(t0 + 1, int(end_frac * n_text))
            for pos in text_positions[t0:t1]:
                instr_mask[pos] = True
    return instr_mask, text_positions, prompt_text, char_span


def _get_spatial_merge_size(model: Any) -> int:
    vision_cfg = getattr(model.config, "vision_config", None)
    if vision_cfg is not None:
        return int(getattr(vision_cfg, "spatial_merge_size", 2))
    return 2


def _llm_grid_shape(grid_thw: torch.Tensor, spatial_merge_size: int) -> tuple[int, int, int]:
    t, h, w = [int(x) for x in grid_thw.tolist()]
    return t, h // spatial_merge_size, w // spatial_merge_size


def _vision_heatmaps_for_image(
    attn_vec: np.ndarray,
    vision_positions: np.ndarray,
    grid_thw: torch.Tensor,
    offset: int,
    spatial_merge_size: int = 2,
) -> tuple[np.ndarray, int]:
    t, llm_h, llm_w = _llm_grid_shape(grid_thw, spatial_merge_size)
    n_tokens = t * llm_h * llm_w
    img_positions = vision_positions[offset : offset + n_tokens]
    if len(img_positions) != n_tokens:
        raise ValueError(
            f"Vision token count mismatch: expected {n_tokens} "
            f"(grid={grid_thw.tolist()}, merge={spatial_merge_size}), "
            f"got {len(img_positions)} at offset {offset}"
        )
    weights = attn_vec[img_positions]
    heatmap = weights.reshape(t, llm_h, llm_w).mean(axis=0)
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max()
    if denom > 0:
        heatmap = heatmap / denom
    return heatmap.astype(np.float32), offset + n_tokens


def extract_s2_attentions(
    model: Any,
    inputs: Any,
    output_ids: torch.Tensor,
    tokenizer: Any,
    llm_output: str,
    instruction: str = "",
    layers: Sequence[int] | None = None,
) -> S2AttentionBundle:
    """Run a teacher-forcing forward pass and aggregate text/vision attention."""
    image_token_id = model.config.image_token_id
    spatial_merge_size = _get_spatial_merge_size(model)
    prompt_len = inputs.input_ids.shape[1]
    full_ids = output_ids[:, : output_ids.shape[1]]

    forward_kwargs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids),
        "pixel_values": inputs.pixel_values,
        "image_grid_thw": inputs.image_grid_thw,
        "output_attentions": True,
        "use_cache": False,
        "return_dict": True,
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        outputs = model(**forward_kwargs)

    if outputs.attentions is None:
        raise RuntimeError(
            "Model returned no attentions. Load the model with attn_implementation='eager'."
        )

    num_layers = len(outputs.attentions)
    layer_indices = _resolve_layer_indices(num_layers, layers)
    selected_attentions = {
        layer_idx: outputs.attentions[layer_idx] for layer_idx in layer_indices
    }
    del outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        # vision: input_ids == image_token_id
        # text is all others
    text_mask, vision_mask = _build_token_masks(full_ids, image_token_id)
    query_positions = _find_query_positions(full_ids, output_ids, prompt_len, tokenizer, llm_output)

    input_token_strings = [
        tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in full_ids[0].tolist()
    ]

    instruction_mass_per_layer: dict[int, float] = {}
    prompt_text_attn_per_layer: dict[int, np.ndarray] = {}
    vision_heatmaps_per_layer: dict[int, list[np.ndarray]] = {}

    instruction_token_mask, prompt_text_positions, prompt_text_string, instruction_char_span = (
        _find_instruction_token_mask(full_ids, text_mask, instruction, tokenizer, prompt_len)
    )

    vision_positions = np.where(vision_mask)[0]
    if isinstance(inputs.image_grid_thw, torch.Tensor) and inputs.image_grid_thw.ndim == 2:
        grid_rows = [inputs.image_grid_thw[i] for i in range(inputs.image_grid_thw.shape[0])]
    else:
        grid_rows = list(inputs.image_grid_thw)

    for layer_idx in layer_indices:
        # (batch, heads, seq, seq) -> mean over heads and query positions

        # (seq_len, seq_len)
        layer_attn = (
            selected_attentions[layer_idx][0].mean(dim=0).detach().float().cpu().numpy()
        )

        # (seq_len, )
        query_attn = layer_attn[query_positions].mean(axis=0)

        instruction_mass = float(query_attn[instruction_token_mask].sum())
        instruction_mass_per_layer[layer_idx] = instruction_mass
        if prompt_text_positions:
            prompt_text_attn_per_layer[layer_idx] = query_attn[prompt_text_positions].astype(np.float32)

        offset = 0
        heatmaps: list[np.ndarray] = []
        for grid_thw in grid_rows:
            heatmap, offset = _vision_heatmaps_for_image(
                query_attn, vision_positions, grid_thw, offset, spatial_merge_size
            )
            heatmaps.append(heatmap)
        vision_heatmaps_per_layer[layer_idx] = heatmaps

    del selected_attentions

    return S2AttentionBundle(
        query_positions=query_positions,
        layer_indices=layer_indices,
        vision_heatmaps_per_layer=vision_heatmaps_per_layer,
        text_token_mask=text_mask,
        vision_token_mask=vision_mask,
        input_token_strings=input_token_strings,
        llm_output=llm_output,
        instruction=instruction,
        instruction_token_mask=instruction_token_mask,
        instruction_mass_per_layer=instruction_mass_per_layer,
        prompt_text_positions=prompt_text_positions,
        prompt_text_string=prompt_text_string,
        instruction_char_span=instruction_char_span,
        prompt_text_attn_per_layer=prompt_text_attn_per_layer,
    )


def _overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = image.convert("RGB")
    heatmap_img = Image.fromarray((plt.cm.jet(heatmap)[..., :3] * 255).astype(np.uint8)).resize(image.size)
    base = np.asarray(image).astype(np.float32)
    overlay = np.asarray(heatmap_img).astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * overlay
    return np.clip(blended, 0, 255).astype(np.uint8)


def _figure_to_rgb(fig) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _instruction_only_series(
    bundle: S2AttentionBundle,
) -> tuple[str, list[int], list[str]] | None:
    """Return instruction text, indices into prompt_text_attn, and token labels."""
    if (
        not bundle.prompt_text_attn_per_layer
        or not bundle.prompt_text_positions
        or bundle.instruction_token_mask is None
    ):
        return None

    pos_to_idx = {pos: idx for idx, pos in enumerate(bundle.prompt_text_positions)}
    instr_indices = [
        pos_to_idx[pos]
        for pos in bundle.prompt_text_positions
        if bundle.instruction_token_mask[pos]
    ]
    if not instr_indices:
        return None

    if bundle.instruction_char_span and bundle.prompt_text_string:
        start, end = bundle.instruction_char_span
        instruction_text = bundle.prompt_text_string[start:end].strip()
    else:
        instruction_text = bundle.instruction.strip()

    token_labels = [
        bundle.input_token_strings[bundle.prompt_text_positions[idx]].replace("\n", " ")
        for idx in instr_indices
    ]
    return instruction_text, instr_indices, token_labels


def save_instruction_only_attention(
    bundle: S2AttentionBundle,
    output_dir: str,
    frame_tag: str,
) -> dict[str, str]:
    """Paper-style figure aligned only to instruction.txt tokens."""
    import os

    series = _instruction_only_series(bundle)
    if series is None:
        return {}

    instruction_text, instr_indices, token_labels = series
    layers = bundle.layer_indices

    attn_rows = []
    for layer in layers:
        row = bundle.prompt_text_attn_per_layer.get(layer)
        if row is None:
            continue
        row = row[instr_indices].astype(np.float32)
        denom = row.max()
        if denom > 0:
            row = row / denom
        attn_rows.append(row)
    if not attn_rows:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    attn_matrix = np.stack(attn_rows, axis=0)
    n_layers = len(attn_rows)

    fig_h = 2.0 + 1.1 * n_layers
    fig_w = max(12, 0.35 * attn_matrix.shape[1])
    fig, axes = plt.subplots(
        n_layers + 1,
        1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1.2] + [1.0] * n_layers},
    )

    ax_text = axes[0]
    ax_text.axis("off")
    ax_text.text(
        0.01,
        0.55,
        instruction_text,
        transform=ax_text.transAxes,
        fontsize=10,
        va="center",
        wrap=True,
        bbox=dict(facecolor="#f4a582", edgecolor="none", alpha=0.35, pad=4),
    )

    for i, layer in enumerate(layers[:n_layers]):
        ax = axes[i + 1]
        im = ax.imshow(
            attn_matrix[i : i + 1],
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            extent=[0, attn_matrix.shape[1], 0, 1],
        )
        ax.set_yticks([0.5])
        ax.set_yticklabels([f"Layer {layer}"])
        if i == n_layers - 1:
            step = max(1, len(token_labels) // 20)
            tick_idx = list(range(0, len(token_labels), step))
            ax.set_xticks([i + 0.5 for i in tick_idx])
            ax.set_xticklabels(
                [token_labels[j][:12] for j in tick_idx],
                rotation=45,
                ha="right",
                fontsize=7,
            )
            ax.set_xlabel("instruction.txt tokens")
        else:
            ax.set_xticks([])
        if i == 0:
            fig.colorbar(im, ax=axes[1:], fraction=0.015, pad=0.01, label="Attention")

    fig.suptitle(
        f"instruction.txt attention only (query pos {bundle.query_positions})",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{frame_tag}_attn_instruction_only.png")
    Image.fromarray(_figure_to_rgb(fig)).save(out_path)
    return {"instruction_only": out_path}


def save_attention_visualizations(
    bundle: S2AttentionBundle,
    input_images: list[Image.Image],
    output_dir: str,
    frame_tag: str,
) -> dict[str, str]:
    """Save instruction and vision attention visualizations."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    saved: dict[str, str] = {}

    saved.update(save_instruction_only_attention(bundle, output_dir, frame_tag))

    layers = bundle.layer_indices

    for layer_idx in layers:
        heatmaps = bundle.vision_heatmaps_per_layer[layer_idx]
        n_images = min(len(heatmaps), len(input_images))
        if n_images == 0:
            continue

        fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
        if n_images == 1:
            axes = [axes]

        for img_i in range(n_images):
            overlay = _overlay_heatmap_on_image(input_images[img_i], heatmaps[img_i])
            axes[img_i].imshow(overlay)
            label = "current" if img_i == n_images - 1 else f"history_{img_i}"
            axes[img_i].set_title(f"Layer {layer_idx} | {label}")
            axes[img_i].axis("off")

        fig.suptitle(f"Vision attention heatmaps — layer {layer_idx}")
        fig.tight_layout()
        heatmap_path = os.path.join(output_dir, f"{frame_tag}_attn_vision_layer{layer_idx:02d}.png")
        Image.fromarray(_figure_to_rgb(fig)).save(heatmap_path)
        saved[f"vision_layer_{layer_idx}"] = heatmap_path

    return saved


def parse_layer_list(layer_str: str | None) -> list[int] | None:
    if not layer_str:
        return None
    return [int(x.strip()) for x in layer_str.split(",") if x.strip()]
