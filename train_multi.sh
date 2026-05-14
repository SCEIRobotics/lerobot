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
  --dataset.streaming=true \
  --dataset.collate_fn=lerobot.policies.flower.utils.FlowerDataCollator \
  --dataset.image_transforms.enable=true \
  --policy.type=flower \
  --policy.training_stage=pretrain \
  --policy.freeze_embeddings_only=true \
  --policy.vlm_path=/mnt/data/share/models/Florence-2-large \
  --policy.horizon=64 \
  --policy.n_action_steps=64 \
  --policy.resize_h=224 \
  --policy.resize_w=224 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.gradient_accumulation_steps=8 \
  --policy.action_spaces='{"joint_single":0, "bimanual":1, "bimanual_nav":2}' \
  --policy.action_dims='{"joint_single":8, "bimanual":14, "bimanual_nav":16}' \
  --policy.state_dims='{"joint_single":8, "bimanual":14, "bimanual_nav":16}' \
  --policy.robot_arm='{"joint_single":1, "bimanual":2, "bimanual_nav":2}' \
  --policy.robot_mapping='{
    "franka":0,
    "aloha": 1, 
    "lift2": 1, 
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