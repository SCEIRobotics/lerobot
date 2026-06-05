#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl
# export EGL_DEVICE_ID=0
export HF_HOME=/mnt/data/cache/huggingface
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export TOKENIZERS_PARALLELISM=false


REPO_ID_LIST=()
ROOT_LIST=()
DATA_DIR_1=/mnt/data/share/datasets/InternRobotics/InternData-A1/interna1_merge_all #

select_dataset=(
    interna1_franka_processed_diff_merge
    interna1_franka_processed_same_merge
    interna1_genie1_processed_merge
    interna1_lift2_processed_diff
    interna1_lift2_processed_same_merge
    interna1_split_aloha_processed_merge
)

for dir in "$DATA_DIR_1"/*/; do
    folder_name=$(basename "${dir%/}")
    if [[ ! " ${select_dataset[@]} " =~ " ${folder_name} " ]]; then
        continue
    fi
    ROOT_LIST+=("${DATA_DIR_1}/${folder_name}")
    REPO_ID_LIST+=("dual/${folder_name}")
done


joined_repo_id=$(printf '"%s",' "${REPO_ID_LIST[@]}" | sed 's/,$//')
repo_ids='['"$joined_repo_id"']'

joined_root=$(printf '"%s",' "${ROOT_LIST[@]}" | sed 's/,$//')
roots='['"$joined_root"']'


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id="${repo_ids}" \
  --dataset.root="${roots}" \
  --dataset.image_transforms.enable=true \
  --dataset.use_shard=true \
  --dataset.keep_in_memory=true \
  --dataset.load_columns="['observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'index', 'task_index']" \
  --dataset.resize='["224", "224"]' \
  --policy.type=flower \
  --policy.training_stage=pretrain \
  --policy.freeze_embeddings_only=true \
  --policy.vlm_path=/mnt/data/share/models/Florence-2-large \
  --policy.horizon=64 \
  --policy.n_action_steps=64 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.gradient_accumulation_steps=2 \
  --policy.action_spaces='{"padding":0}' \
  --policy.action_dims='{"padding":32}' \
  --policy.state_dims='{"padding":32}' \
  --policy.robot_arm='{"padding":2}' \
  --policy.robot_mapping='{
    "franka":0,
    "aloha": 0, 
    "lift2": 0, 
    "genie1": 0,
    }' \
  --policy.robot_action_dim='{
    "franka":8,
    "aloha": 14, 
    "lift2": 14, 
    "genie1": 16,
    }' \
  --policy.robot_num_arms='{
    "franka":1,
    "aloha": 2, 
    "lift2": 2, 
    "genie1": 2,
    }' \
  --batch_size=64 \
  --num_workers=2 \
  --steps=1600000 \
  --save_freq=40000 \
  --output_dir=./outputs/pretrain/dual-${TIMESTAMP} \
  --job_name=dual-${TIMESTAMP} \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --wandb.mode=offline \
  --wandb.project=lerobot-pretrain \