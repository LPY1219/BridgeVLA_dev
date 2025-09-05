#!/bin/bash

huggingface-cli download --repo-type dataset\
 --resume-download "XiangnanW/BridgeVLA_Dev" \
 --local-dir "/home/lpy/BridgeVLA_dev/finetune/Real/test_data"\
 --local-dir-use-symlinks False