#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl
# export EGL_DEVICE_ID=0

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export TOKENIZERS_PARALLELISM=false
# export WANDB_MODE=offline

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
accelerate launch \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=lerobot/libero_10_lerobot \
  --dataset.root=/mnt/data_ssd/share/datasets/flower/hf_dataset/libero \
  --dataset.collate_fn=lerobot.policies.flower.utils.FlowerDataCollator \
  --dataset.collate_fn_params='{"vlm_path": "/mnt/data_ssd/share/models/Florence-2-large"}' \
  --policy.type=flower \
  --policy.training_stage=sft \
  --policy.freeze_embeddings_only=false \
  --policy.load_pretrained=true \
  --policy.pretrained_model_path=/mnt/data_ssd/share/models/flower_vla_pret/360000_model_weights.pt \
  --policy.vlm_path=/mnt/data_ssd/share/models/Florence-2-large \
  --policy.horizon=10 \
  --policy.n_action_steps=10 \
  --policy.resize_h=112 \
  --policy.resize_w=112 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --num_workers=4 \
  --steps=100000 \
  --save_freq=10000 \
  --output_dir=./outputs-1/train-libero-flower-libero10-${TIMESTAMP} \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --wandb.mode=offline \
  --wandb.project=lerobot-libero-refactor \
#   --eval_freq=10000 \
#   --eval.n_episodes=50 \
#   --eval.batch_size=2 \
#   --env.type=libero \
#   --env.task=libero_10 \

