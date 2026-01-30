#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

# export MUJOCO_GL=egl # 强制 MuJoCo 使用 EGL 渲染（关键）
# export PYOPENGL_PLATFORM=egl # 禁用 GLFW 图形窗口（避免初始化错误）
# export EGL_DEVICE_ID=0

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline  # 强制离线记录, 但是lerobot默认是online的, 需要设置wandb.mode=offline
export WANDB_API_KEY=7a17221f579b43949e05faf2a9120c5a6b6506e5
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# accelerate launch --multi_gpu --num_processes=2 \
python ./src/lerobot/scripts/lerobot_train_multi.py \
    --dataset.root=/mnt/data_ssd/share \
    --dataset.root_val=/mnt/data_ssd/share \
    --dataset.repo_id=InternData-A1/test \
    --dataset.streaming=true \
    --dataset.requires_padding=true \
    --policy.type=flower \
    --policy.n_obs_steps=1 \
    --policy.horizon=64 \
    --policy.n_action_steps=60 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --batch_size=32 \
    --num_workers=4 \
    --steps=800000 \
    --save_freq=20000 \
    --valid_freq=10 \
    --output_dir=outputs/train/train-a1-${TIMESTAMP} \
    --wandb.enable=false \
    --wandb.disable_artifact=true \
    --wandb.mode=offline \