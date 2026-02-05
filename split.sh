#! /bin/bash

lerobot-edit-dataset \
    --repo_id interna1_merge_all/interna1_franka_processed_same_merge \
    --root /vla-cd/interna1_merge_all/interna1_franka_processed_same_merge \
    --root_1 /vla-cd/interna1_merge_all/ \
    --operation.type split \
    --operation.splits '{"interna1_franka_processed_same_merge_val": [0,100,200]}'