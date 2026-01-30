#! /bin/bash

lerobot-edit-dataset \
    --repo_id datasets/aloha_sim_transfer_cube_scripted \
    --root /mnt/data_ssd/share/datasets/aloha_sim_transfer_cube_scripted \
    --root_1 /mnt/data_ssd/share/datasets \
    --operation.type split \
    --operation.splits '{"aloha_sim_transfer_cube_scripted_train": 0.8, "aloha_sim_transfer_cube_scripted_val": 0.2}'