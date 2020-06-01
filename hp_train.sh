#!/bin/bash
BATCH_SIZE=8
WORKER_NUMBER=8
LEARNING_RATE=0.00001

# -mpath  $CHECK_POINT

# todo: why bash script doesn't work

GPU_ID=6,7

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -m MNet \
-bs $BATCH_SIZE -lr $LEARNING_RATE -nw $WORKER_NUMBER -mGPUs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py -m MNet \
-r True -checkpoint checkpoint/MNet_epoch_580 \
-bs 1024 -lr 0.00001 -nw 256 -mGPUs

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py -m AF3 \
-mpath checkpoint/MNet_epoch_995 \
-bs 128 -lr 0.1 -nw 16 -mGPUs