#!/bin/bash

# chmod +x train_jetson.sh
# ./train_jetson.sh
set -e

# Single Jetson / local training config
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# DeepSpeed config
deepspeed=scripts/train/qwenvl_train/zero2.json

# Training config
lr=1e-4
batch_size=1
grad_accum_steps=2
max_pixels=313600
min_pixels=3136

# vln_datasets=r2r_125cm_0_30%30,r2r_60cm_15_15%30,rxr_125cm_0_30%30,rxr_60cm_15_15%30,scalevln_125cm_0_30%30,scalevln_60cm_30_30%30
vln_datasets=r2r_125cm_0_30%30,r2r_60cm_15_15%30,rxr_125cm_0_30%30,rxr_60cm_15_15%30,scalevln_125cm_0_30%30,scalevln_60cm_30_30%30

run_name=InternVLA-N1-DualVLN-Jetson
output_dir=checkpoints/${run_name}

system1=navdp_async
system2_ckpt=checkpoints/InternVLA-N1-System2

torchrun --nnodes=1 --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${system2_ckpt}" \
    --vln_dataset_use ${vln_datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --bf16 \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only True \
    --system1 ${system1} \
    --output_dir ${output_dir} \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1e-05}' \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name ${run_name} \
    --report_to none
