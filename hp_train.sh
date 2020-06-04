#!/bin/bash
BATCH_SIZE=1024
WORKER_NUMBER=256
LEARNING_RATE=0.1

GPU_ID=4,5,6,7

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -m MNet -r True -checkpoint checkpoint/MNet_epoch_65 -bs $BATCH_SIZE -lr $LEARNING_RATE -nw $WORKER_NUMBER -mGPUs
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -m MNet -r True -checkpoint checkpoint/MNet_epoch_65 -bs 1024 -lr 0.1 -nw 256 -mGPUs

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py -m AF3 -mpath checkpoint/MNet_epoch_65 -bs 128 -lr 0.1 -nw 32 -mGPUs
