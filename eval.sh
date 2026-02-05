#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false


export MUJOCO_GL=egl # 强制 MuJoCo 使用 EGL 渲染（关键）
export PYOPENGL_PLATFORM=egl # 禁用 GLFW 图形窗口（避免初始化错误）
export EGL_DEVICE_ID=0


lerobot-eval \
  --policy.path="" \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=50
