#!/bin/bash
cd /workspace/STDev
M768='--override LEVEL_CHANNELS=(768,512,1024) --override DECODER_LEVEL_BLOCKS=(8,2,2)'
M1024='--override LEVEL_CHANNELS=(1024,512,1024) --override DECODER_LEVEL_BLOCKS=(8,2,2)'
launch() {
    gpu=$1; name=$2; overrides=$3; latent_size=$4; shift 4
    (
        for spec in "$@"; do
            batch_size=${spec%:*}; checkpointing=${spec#*:}
            CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python dev/pretrain_training_throughput.py --name $name $overrides \
                --latent-sizes $latent_size --batch-sizes $batch_size --checkpointing $checkpointing --steps 15 \
                >> dev/temporary/throughput/third_$gpu.log 2>&1
        done
    ) &
}
launch 0 768_dec8 "$M768" 128 4:0
launch 1 768_dec8 "$M768" 128 8:1
launch 2 768_dec8 "$M768" 128 2:0
launch 3 1024_dec8 "$M1024" 64 8:0
launch 4 1024_dec8 "$M1024" 256 1:1
launch 5 768_dec8 "$M768" 64 16:0
wait
