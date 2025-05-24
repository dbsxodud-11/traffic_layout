#!/usr/bin/env bash

WAITING_TIME=(100 300 500 700 900 1100 1300)
# WAITING_TIME=(100)
GUIDE_TEMP=(0.1 0.25 0.5)
NUM=(500)

for waiting_time in "${WAITING_TIME[@]}"; do
    for guide_temp in "${GUIDE_TEMP[@]}"; do
        for num in "${NUM[@]}"; do
            echo "Running with waiting_time='$waiting_time' and guide_temp='$guide_temp' and num='$num'"
            CUDA_VISIBLE_DEVICES=5 python scripts/generate.py \
                -n $num \
                -c config_files/generation_defaults.yaml \
                -p "waiting_time=$waiting_time" \
                -o "sampler.guide_temp=$guide_temp"
        done
    done
done
