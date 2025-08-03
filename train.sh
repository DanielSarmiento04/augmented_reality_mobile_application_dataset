#!/bin/bash

# --- Default values ---
DEFAULT_MODEL="yolov11n.pt" # Changed default to yolov11n as yolov11n is not standard
DEFAULT_DATA="./train/custom.yml"
DEFAULT_EPOCHS=100
DEFAULT_IMGSZ=640
DEFAULT_DEVICE=0
DEFAULT_BATCH=-1

# --- Assign arguments to variables, using defaults if not provided ---
# Usage: ./train.sh [model] [data_yaml] [epochs] [imgsz] [device] [batch]
MODEL=${1:-$DEFAULT_MODEL}
DATA_YAML=${2:-$DEFAULT_DATA}
EPOCHS=${3:-$DEFAULT_EPOCHS}
IMGSZ=${4:-$DEFAULT_IMGSZ}
DEVICE=${5:-$DEFAULT_DEVICE}
BATCH=${6:-$DEFAULT_BATCH}

# --- Login to wandb (optional, keep if needed) ---
wandb login

# --- Run YOLO training ---
echo "Starting YOLO training with the following parameters:"
echo "Model: $MODEL"
echo "Data YAML: $DATA_YAML"
echo "Epochs: $EPOCHS"
echo "Image Size: $IMGSZ"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH"

yolo task=detect mode=train model=$MODEL data=$DATA_YAML epochs=$EPOCHS imgsz=$IMGSZ plots=True device=$DEVICE batch=$BATCH