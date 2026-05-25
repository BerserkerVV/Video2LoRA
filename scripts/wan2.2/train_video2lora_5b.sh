#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-models/Diffusion_Transformer/Wan2.2-TI2V-5B-Diffusers}"
export DATASET_NAME="${DATASET_NAME:-datasets/video_as_prompt}"
export DATASET_META_NAME="${DATASET_META_NAME:-datasets/video_as_prompt/metadata.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/wan2.2-ti2v-5b-video2lora}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_lora.py \
  --enable_video2lora \
  --config_path="config/wan2.2/wan_diffusers_5b.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --image_sample_size=512 \
  --video_sample_size=256 \
  --token_sample_size=256 \
  --video_sample_stride=2 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --max_train_steps="${MAX_TRAIN_STEPS:-1000}" \
  --checkpointing_steps="${CHECKPOINTING_STEPS:-100}" \
  --learning_rate="${LEARNING_RATE:-1e-4}" \
  --rank="${LILORA_RANK:-1}" \
  --network_alpha="${LILORA_ALPHA:-1}" \
  --down_dim="${LILORA_DOWN_DIM:-200}" \
  --up_dim="${LILORA_UP_DIM:-100}" \
  --seed="${SEED:-42}" \
  --output_dir="$OUTPUT_DIR" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="full" \
  --train_mode="ti2v" \
  --low_vram
