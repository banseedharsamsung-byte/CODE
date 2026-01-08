@echo off
REM Multi-GPU training script for Windows using accelerate
REM Usage: train_multi_gpu.bat [num_gpus]

set NUM_GPUS=%1
if "%NUM_GPUS%"=="" set NUM_GPUS=4

echo Starting multi-GPU training with %NUM_GPUS% GPUs...

REM Launch training with accelerate
accelerate launch --num_processes=%NUM_GPUS% --mixed_precision=fp16 train.py

