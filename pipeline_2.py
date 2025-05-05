# %%
import os
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm
import argparse
import shutil # Added for copying files

# %%
# Set up argument parser
parser = argparse.ArgumentParser(description='Augment image dataset with YOLO labels.')
parser.add_argument('--input_images_dir', type=str, required=True, help='Path to the input images directory.')
parser.add_argument('--input_labels_dir', type=str, required=True, help='Path to the input labels directory.')
parser.add_argument('--output_images_dir', type=str, required=True, help='Path to the output augmented images directory.')
parser.add_argument('--output_labels_dir', type=str, required=True, help='Path to the output augmented labels directory.')
parser.add_argument('--classes_file', type=str, default='./classes.txt', help='Path to the file containing class names.')
parser.add_argument('--num_augmentations', type=int, default=20, help='Number of augmentations to generate per image.')

# Parse arguments
args = parser.parse_args()

# %%
# Use parsed arguments for paths and settings
input_images_dir = args.input_images_dir
input_labels_dir = args.input_labels_dir
output_images_dir = args.output_images_dir
output_labels_dir = args.output_labels_dir
classes_file_path = args.classes_file
num_augmentations = args.num_augmentations

# %%
# Read classes from the specified file
try:
    with open(classes_file_path) as f:
        classes = f.read().splitlines()
except FileNotFoundError:
    print(f"Error: Classes file not found at {classes_file_path}")
    exit(1) # Exit if classes file is essential and not found

# %%
# Function to read YOLO-formatted label file
def read_yolo_label(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    bboxes = []
    class_labels = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(int(class_id))
    return bboxes, class_labels

# Function to save YOLO-formatted label file
def save_yolo_label(label_path, bboxes, class_labels):
    with open(label_path, 'w') as file:
        for bbox, class_id in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            file.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# %%
# Ensure output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# %%
# Define the augmentation pipeline
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        A.BBoxSafeRandomCrop(erosion_rate=0.0, p=0.6),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.6),
        A.OpticalDistortion(p=0.6),
        A.FrequencyMasking(p=0.6),
        A.GaussNoise(p=0.6),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

# %%
# Process each image and its corresponding label

# Get list of images to process
try:
    image_files = [f for f in os.listdir(input_images_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]
except FileNotFoundError:
    print(f"Error: Input images directory not found at {input_images_dir}")
    exit(1)

for image_name in tqdm(image_files):
    image_path = os.path.join(input_images_dir, image_name)
    label_name = os.path.splitext(image_name)[0] + '.txt'
    label_path = os.path.join(input_labels_dir, label_name)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image file {image_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Read labels
    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} does not exist.")
        continue
    bboxes, class_labels = read_yolo_label(label_path)
    
    # Apply augmentations num_augmentations times
    for i in range(num_augmentations): # Use parsed argument
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']

            # Generate new file names
            base_name = os.path.splitext(image_name)[0]
            new_image_name = f"{base_name}_aug_{i}.jpg"
            new_label_name = f"{base_name}_aug_{i}.txt"

            # Save augmented image
            transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(output_images_dir, new_image_name), 
                transformed_image_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            )

            # Save augmented label
            save_yolo_label(os.path.join(output_labels_dir, new_label_name), transformed_bboxes, transformed_class_labels)

        except Exception as e:
            continue

# %%
# Copy original images and labels to the output directories
print("Copying original files to output directories...")

# List files again after augmentation loop, in case new files were added (though unlikely here)
try:
    input_image_files = [f for f in os.listdir(input_images_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]
    input_label_files = [f for f in os.listdir(input_labels_dir) if f.endswith('.txt')]
except FileNotFoundError:
    print(f"Error: Input directory not found during final copy step.")
    exit(1)

for image_file in tqdm(input_image_files):
    src_image_path = os.path.join(input_images_dir, image_file)
    dst_image_path = os.path.join(output_images_dir, image_file)
    try:
        shutil.copy2(src_image_path, dst_image_path) # copy2 preserves metadata
    except Exception as e:
        print(f"Warning: Could not copy image {image_file}: {e}")

for label_file in tqdm(input_label_files):
    src_label_path = os.path.join(input_labels_dir, label_file)
    dst_label_path = os.path.join(output_labels_dir, label_file)
    # Check if the corresponding image exists before copying label
    corresponding_image_base = os.path.splitext(label_file)[0]
    image_exists = any(corresponding_image_base == os.path.splitext(img)[0] for img in input_image_files)
    if image_exists and os.path.exists(src_label_path):
         try:
            shutil.copy2(src_label_path, dst_label_path) # copy2 preserves metadata
         except Exception as e:
            print(f"Warning: Could not copy label {label_file}: {e}")
    elif not image_exists:
         print(f"Warning: Skipping label {label_file} as corresponding image was not found or processed.")

print("Original files copied.")
print("Dataset augmentation and copying completed.")



