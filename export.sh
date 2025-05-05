#!/bin/bash

# Set default model path
DEFAULT_MODEL_PATH="best.pt"

# Use the first argument as the model path, or the default if no argument is provided
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

# Export the model
yolo export model="$MODEL_PATH" format=tflite nms=True imgsz=640