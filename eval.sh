
lerobot-eval \
  --policy.path="/mnt/data/daiwanqin/gitlab/lerobot/outputs/train/pick-20251210_001050/checkpoints/170000/pretrained_model" \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=3

# "/mnt/data/daiwanqin/gitlab/lerobot/outputs/train/pick-20251210_001050/checkpoints/170000/pretrained_model"