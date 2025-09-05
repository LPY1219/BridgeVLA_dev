#! /bin/bash

cd /home/lpy/BridgeVLA_dev/finetune/Real

# train 0830
bash my_train.sh --exp_cfg_path "/home/lpy/BridgeVLA_dev/finetune/Real/configs/real.yaml" \
                --mvt_cfg_path "/home/lpy/BridgeVLA_dev/finetune/bridgevla/mvt/configs/rvt2.yaml" \
                --exp_note "train_put_coke_with_different_rotation" \
                --cameras "3rd" \
                --ep_per_task 20 \
                --data_folder "/home/lpy/BridgeVLA_dev/finetune/Real/data" \
                --log_dir "/home/lpy/BridgeVLA_dev/finetune/Real/logs" \
                --freeze_vision_tower

                