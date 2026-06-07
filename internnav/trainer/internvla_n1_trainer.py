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

import qwenvl_base  # noqa: F401  # Trainer/model monkey patches; flash_attn not required unless data_flatten
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2VLImageProcessor,
    Trainer,
    TrainerCallback,
)

import internnav.dataset.internvla_n1_lerobot_dataset as lerobot_dataset
from internnav.dataset.internvla_n1_lerobot_dataset import make_supervised_data_module
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
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
)


class JetsonTrainerCallback(TrainerCallback, JetsonTrainingCallback):
    """Bridge HF TrainerCallback with Jetson training/memory logging."""

    def __init__(self):
        JetsonTrainingCallback.__init__(self)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

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
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def is_rank0() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def set_model(model_args, model):
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
        model = InternVLAN1ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
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

    if data_args.data_flatten:
        from qwenvl_base import replace_qwen2_vl_attention_class

        replace_qwen2_vl_attention_class()
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

    if data_args.model_type == "internvla-n1":
        model.get_model().initialize_vision_modules(model_args=model_args)
    set_model(model_args, model)

    if is_rank0():
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
        print_status_block("after_model_load")

    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)  # noqa: F821
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    train_dataset = data_module["train_dataset"]
    if is_rank0():
        print(f"train_dataset size: {len(train_dataset)}")
        print_status_block("after_dataset_load")
    if len(train_dataset) == 0:
        raise RuntimeError(
            "train_dataset has 0 samples. Check dataset debug output above for path/schema issues."
        )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[JetsonTrainerCallback()],
        **data_module,
    )
    from tabulate import tabulate

    if trainer.is_world_process_zero():
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"trainable params: {trainable_params:,} / {total_params:,}")
        if trainable_params == 0:
            print("WARN: all parameters frozen — training will not update weights")
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
        print_status_block("before_train_loop")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="sdpa")
