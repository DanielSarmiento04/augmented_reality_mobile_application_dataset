# %%
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cv2
import shutil
import argparse  # Add argparse import

# %%
# Setup argument parser
parser = argparse.ArgumentParser(description='Process and quantize images based on labels.')
parser.add_argument('--input_dir', type=str, default='split_dataset',
                    help='Base directory containing the split dataset (train, test, val folders)')
parser.add_argument('--output_dir', type=str, default='quantize/',
                    help='Directory where the quantized output will be saved')

# Parse arguments
args = parser.parse_args()

# Use parsed arguments for paths
base = args.input_dir
output_dir = args.output_dir

# %%
# Update paths based on parsed arguments
test_labels = f'{base}/test/labels/'
train_labels = f'{base}/train/labels/'
val_labels = f'{base}/val/labels/'

test_images = f'{base}/test/images/'
train_images = f'{base}/train/images/'
val_images = f'{base}/val/images/'

# %%
list_test_images = os.listdir(test_images)
list_train_images = os.listdir(train_images)
list_val_images = os.listdir(val_images)

# %%
# Ensure output directory exists using the parsed argument
os.makedirs(output_dir, exist_ok=True)

# %%
# make a dictionary of the labels id 0- 80 and put 0 value in them
labels_count = {
    i: 0 for i in range(84)
}
labels_count

# %%
labels_file_path = [test_labels, train_labels, val_labels]

idx = 80

images_extensions = ['jpg', 'jpeg', 'png', 'JPG']

for path in labels_file_path:
    
    category = path.split('/')[-3]
    # Use parsed output_dir
    os.makedirs(f'{output_dir}{category}/labels', exist_ok=True)
    os.makedirs(f'{output_dir}{category}/images', exist_ok=True)

    for file in os.listdir(path):
        with open(path + file, 'r') as f:
            lines = f.readlines()
            # check
            is_pump = list(
                filter(lambda x: x.split()[0] == str(idx), lines)
            )
            if len(is_pump) > 0:
                # copy to output in category folder using parsed output_dir
                with open(f'{output_dir}{category}/labels/{file}', 'w') as f:
                    f.writelines(lines)
                
                # copy image to output in category folder
                # Use parsed input_dir (base)
                input_base_dir = path.split('/')[0]  # This should now correctly reference args.input_dir
                for ext in images_extensions:
                    # print(file)
                    input_image = f'{input_base_dir}/{category}/images/{file.replace("txt", ext)}'
                    # print(input_image)
                    if os.path.exists(input_image):  # Check existence using the correct input path
                        # Use parsed output_dir
                        output_image = f'{output_dir}{category}/images/{file.replace("txt", ext)}'
                        print(input_image, output_image)

                        # reduce quality image
                        img = cv2.imread(input_image)

                        height, width = img.shape[:2]
                        new_height = 640
                        new_width = int((new_height / height) * width)

                        image_resize = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                        _, encoded_img = cv2.imencode('.jpg', image_resize, encode_param)
                            
                        # Decode the compressed image
                        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

                        # save image
                        cv2.imwrite(output_image, decoded_img)

                        break



