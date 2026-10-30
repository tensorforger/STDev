#!/bin/bash
cd /workspace/STDev
M1024='--override LEVEL_CHANNELS=(1024,512,1024) --override DECODER_LEVEL_BLOCKS=(8,2,2)'
M1280='--override LEVEL_CHANNELS=(1280,640,1280) --override DECODER_LEVEL_BLOCKS=(8,2,2)'
launch() {
    gpu=$1; name=$2; overrides=$3; shift 3
    (
        for spec in "$@"; do
            batch_size=${spec%:*}; checkpointing=${spec#*:}
            CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python dev/pretrain_training_throughput.py --name $name $overrides \
                --latent-sizes 128 --batch-sizes $batch_size --checkpointing $checkpointing --steps 15 \
                >> dev/temporary/throughput/second_$gpu.log 2>&1
        done
    ) &
}
launch 0 0078 "" 8:0 4:0
launch 1 1024_dec8 "$M1024" 8:1 12:1
launch 2 1024_dec8 "$M1024" 4:1 2:1
launch 3 1024_dec8 "$M1024" 4:0 2:0
launch 4 1280_dec8 "$M1280" 8:1 12:1
launch 5 1280_dec8 "$M1280" 4:1 2:1
launch 6 1280_dec8 "$M1280" 2:0 1:0
launch 7 0078 "" 24:1 32:1
wait
