# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
flow:
  1. Parse CLI args into ModelArguments, DataArguments, TrainingArguments
  2. Build image transforms (optional augmentation + resize).
  3. Load model based on checkpoint name.
  4. Configure trainable submodules via set_model() (S2 vision/LLM + S1 trajectory head).
  5. Build dataset + collator via make_supervised_data_module()
  6. Run HF Trainer (or a forward-only frozen smoke test when all params are frozen).
  7. Save final weights, trainer state, and image processor to output_dir.

"""

import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Dict

import torch
import transformers
from torchvision.transforms import v2

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Side-effect import: registers Trainer/model monkey patches in qwenvl_base.py.
# replace_qwen2_vl_attention_class() is called later when data_flatten=True.
import qwenvl_base  # noqa: F401
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2VLImageProcessor,
    Trainer,
)

import internnav.dataset.internvla_n1_lerobot_dataset as lerobot_dataset
from internnav.dataset.internvla_n1_lerobot_dataset import make_supervised_data_module
from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    InternVLAN1ForCausalLM,
    InternVLAN1ModelConfig,
)
from internnav.trainer.internvla_n1_argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from internnav.trainer.jetson_monitor import (
    JetsonTrainingCallback,
    format_memory_line,
    print_jetson_summary,
    print_status_block,
    train_tensorboard_log_dir,
)


# Alias kept for backwards compatibility; JetsonTrainingCallback extends TrainerCallback.
JetsonTrainerCallback = JetsonTrainingCallback


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save model weights after training.
    DeepSpeed manages its own sharded state, delegate to trainer.save_model().
    Otherwise gather weights to CPU first to avoid GPU OOM on large checkpoints.
    """

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Add special tokens and grow embedding table; init new rows from existing mean."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def is_rank0() -> bool:
    """True on the primary process in distributed training (rank 0 prints/logs)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def run_frozen_smoke_test(trainer: Trainer) -> None:
    """Forward-only sanity check when every parameter has requires_grad=False.

    Runs a few dataloader batches through compute_loss() without backprop orweight updates.
    """
    trainer.model.eval()
    dataloader = trainer.get_train_dataloader()
    max_steps = trainer.args.max_steps if trainer.args.max_steps and trainer.args.max_steps > 0 else 1

    if trainer.is_world_process_zero():
        print(f"[frozen smoke] running {max_steps} forward pass(es), weights will not change")

    completed = 0
    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break
        batch = trainer._prepare_inputs(batch)
        with torch.no_grad():
            loss = trainer.compute_loss(trainer.model, batch)
        completed = step + 1
        if trainer.is_world_process_zero():
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
            print(f"[frozen smoke] step {completed}/{max_steps} loss={loss_val:.4f}")
            print_status_block(f"frozen_smoke_step_{completed}")

    if trainer.is_world_process_zero():
        print(f"[frozen smoke] done — {completed} forward pass(es), no weight updates")


def set_model(model_args, model):
    """Set requires_grad on submodules according to CLI arg

    S2:
      tune_mm_vision: vision encoder (model.visual)
      tune_mm_mlp: vision-language projector (model.visual.merger)
      tune_mm_llm: transformer + lm_head (model.model, model.lm_head)

    S1 (trajectory head, when system1 is set):
      nextdit: traj_dit, action encoder/decoder, cond_projector, latent_queries, ...
      navdp: navdp policy + latent_queries (rgb_model stays frozen)
    """
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        # model.lm_head.requires_grad = False
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = False

    if 'nextdit' in model_args.system1:
        modules = [
            'action_encoder',
            'action_decoder',
            'traj_dit',
            'cond_projector',
            'memory_encoder',
            'rgb_resampler',
            'rgb_model',
        ]
        for n, p in model.model.named_parameters():
            if any(k in n for k in modules):
                p.requires_grad = True
        model.model.latent_queries.requires_grad = True
    elif 'navdp' in model_args.system1:
        for n, p in model.model.navdp.named_parameters():
            if "rgb_model" not in n:
                p.requires_grad = True
        model.model.latent_queries.requires_grad = True


def train(attn_implementation="sdpa"):
    global local_rank


    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    lerobot_dataset.local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)


    if is_rank0():
        print_jetson_summary()
        print("\n=== Training data env ===")
        print(f"vln_dataset_use: {data_args.vln_dataset_use}")
        print(f"INTERNAV_R2R_DATA_PATH: {os.environ.get('INTERNAV_R2R_DATA_PATH', '(not set)')}")
        print(f"INTERNAV_RXR_DATA_PATH: {os.environ.get('INTERNAV_RXR_DATA_PATH', '(not set)')}")
        print(f"INTERNAV_SCALEVLN_DATA_PATH: {os.environ.get('INTERNAV_SCALEVLN_DATA_PATH', '(not set)')}")
        print("=== End training data env ===\n")

    # Resize is always applied; extra color/sharpness aug when data_augmentation=True.
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


    if 'internvla-n1-system2' in model_args.model_name_or_path.lower():
        internvla_config = InternVLAN1ModelConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        # System2 checkpoints ship with nextdit in config. Defer System1 construction
        # until after checkpoint load — ZeRO-3 partitions params during from_pretrained
        # and breaks NavDP backbone weight loading if S1 modules are built too early.
        if model_args.system1 and model_args.system1 != "none":
            if internvla_config.system1 != "none":
                if is_rank0():
                    print(
                        f"Deferring System1 init during load "
                        f"(checkpoint system1={internvla_config.system1!r}, "
                        f"train system1={model_args.system1!r})"
                    )
                internvla_config.system1 = "none"
        model = InternVLAN1ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=internvla_config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=True,
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "internvla-n1"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=True,
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=True,
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    # --- Memory / attention optimizations ---
    # data_flatten swaps in a flattened attention kernel; disable KV cache during training.
    if data_args.data_flatten:
        from qwenvl_base import replace_qwen2_vl_attention_class

        replace_qwen2_vl_attention_class()
    # use_cache=False is required for gradient checkpointing (no stored past key/values).
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Lazy-build S1 modules after S2 weights are loaded
    if data_args.model_type == "internvla-n1":
        model.get_model().initialize_vision_modules(model_args=model_args)
    set_model(model_args, model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_smoke = trainable_params == 0
    # DeepSpeed needs an optimizer; skip it when running forward-only smoke test.
    if frozen_smoke and training_args.deepspeed:
        if is_rank0():
            print("All parameters frozen — disabling DeepSpeed; forward-only smoke test (no optimizer updates).")
        training_args.deepspeed = None

    if is_rank0():
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
        print_status_block("after_model_load")

    # data_packing=True uses a packed-batch collator (make_supervised_data_module_packed).
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)  # noqa: F821
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    train_dataset = data_module["train_dataset"]
    if is_rank0():
        print(f"train_dataset size: {len(train_dataset)}")
        print(f"TensorBoard:     {train_tensorboard_log_dir(training_args)}")
        print_status_block("after_dataset_load")
    if len(train_dataset) == 0:
        raise RuntimeError(
            "train_dataset has 0 samples. Check dataset debug output above for path/schema issues."
        )


    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[JetsonTrainingCallback()],
        **data_module,
    )
    from tabulate import tabulate

    if trainer.is_world_process_zero():
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"trainable params: {trainable_params:,} / {total_params:,}")
        if trainable_params == 0:
            print("Mode: frozen smoke test (forward only, no checkpoint save)")
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
        print_status_block("before_train_loop")

    if frozen_smoke:
        run_frozen_smoke_test(trainer)
    elif list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
        trainer.save_state()
        data_args.image_processor.save_pretrained(training_args.output_dir)
        model.config.use_cache = True
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    else:
        trainer.train()
        trainer.save_state()
        data_args.image_processor.save_pretrained(training_args.output_dir)
        model.config.use_cache = True
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="sdpa")
