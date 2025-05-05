# %%
import numpy as np
import os
import pandas as pd
import cv2
import json
import matplotlib.pyplot as plt
from pprint import pprint
import argparse  # Add argparse import

# %%
# Set up argument parser
parser = argparse.ArgumentParser(description='Convert COCO annotations to YOLO format with filtering.')
parser.add_argument('--image_path', type=str, required=True, help='Path to the COCO images directory.')
parser.add_argument('--annotation_path', type=str, required=True, help='Path to the COCO annotations directory.')
parser.add_argument('--output_images', type=str, default='./images/', help='Directory to save filtered images.')
parser.add_argument('--output_labels', type=str, default='./labels/', help='Directory to save YOLO annotations.')

# Parse arguments from command line
args = parser.parse_args()

# Use arguments
image_path = args.image_path
annotation_path = args.annotation_path
output_images = args.output_images
output_labels = args.output_labels

# Construct the full path to the annotation file
annotation_file = os.path.join(annotation_path, "instances_train2017.json")

# Create output directories if they don't exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# %%
with open(f'{annotation_path}instances_train2017.json') as f:
    data_2 = json.load(f)

# %%
with open(annotation_file) as f:
    data = json.load(f)

data

# %%
coco_images = data.get('images')
coco_info = data.get('info')
coco_licenses = data.get('licenses')
coco_annotation = data.get('annotations')
coco_categories = data.get('categories')

# %% [markdown]
# ## Cual es el problema
#
# Dentro del d
#

# %%
ids = list(
    map(
        lambda x: (x['id'], x['name']),
        coco_categories
    )
)
ids

# %%
images_ids = list(
    map(
        lambda x: x['category_id'],
        coco_annotation
    )
)

unique, counts = np.unique(images_ids, return_counts=True)

mix_dict = dict(
    zip(
        unique.tolist(),
        counts.tolist()
    )
)

unique

# %%
print("Number of coco_categories: ", len(coco_categories))

# %%
labels = list(
    map(
        lambda x: x.get('name'),
        coco_categories
    )
)
print(len(labels))

# %%
with open("coco_labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

# %%
len(coco_images), \
    len(coco_info), \
    len(coco_licenses), \
    len(coco_annotation), \
    len(coco_categories)

# %%
list(
    filter(
        lambda x: x['id'] == 5,
        coco_categories
    )
)

# %%
MAX_IMAGES_NO_LABEL = 40_000
MAX_IMAGES_PERSON = 10_000
MAX_IMAGES_REST_OF_CATEGORIES = 40_000

# %%
import json
import os


def convert_bbox_coco_to_yolo(image_width, image_height, bbox):
    """
    Convert COCO bounding box format to YOLO format.
    COCO format: [x_min, y_min, width, height]
    YOLO format: [x_center, y_center, width, height] (normalized)
    """
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    # Normalize coordinates
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return [x_center, y_center, width, height]


def coco_to_yolo(coco_json_path, output_dir):
    """
    Convert COCO annotations to YOLO format.

    :param coco_json_path: Path to the COCO JSON file.
    :param output_dir: Directory to save YOLO annotations.
    """
    # Load COCO JSON annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract image dimensions and annotations
    images = {image["id"]: image for image in coco_data["images"]}
    annotations = coco_data["annotations"]

    # Process each annotation
    yolo_annotations = {}
    for ann in annotations:
        image_id = ann["image_id"]
        category_id = ann["category_id"] - 1  # YOLO expects 0-based indexing for class IDs
        bbox = ann["bbox"]

        # Get image dimensions
        image_info = images[image_id]
        image_width = image_info["width"]
        image_height = image_info["height"]

        # Convert bounding box to YOLO format
        yolo_bbox = convert_bbox_coco_to_yolo(image_width, image_height, bbox)

        # Add to annotations for this image
        if image_id not in yolo_annotations:
            yolo_annotations[image_id] = []
        yolo_annotations[image_id].append([category_id, *yolo_bbox])

    # Write YOLO annotations to text files
    for image_id, annotations in yolo_annotations.items():
        image_info = images[image_id]
        image_filename = os.path.splitext(image_info["file_name"])[0]
        output_path = os.path.join(output_dir, f"{image_filename}.txt")

        with open(output_path, 'w') as f:
            for ann in annotations:
                f.write(" ".join(map(str, ann)) + "\n")


# %%
for image_coco in coco_images[:4]:

    image_id = image_coco.get('id')
    image_file_name = image_coco.get('file_name')
    image_license = image_coco.get('license')
    image_coco_url = image_coco.get('coco_url')
    image_date_captured = image_coco.get('date_captured')
    image_flickr_url = image_coco.get('flickr_url')

    annotations = list(
        filter(
            lambda x: x['image_id'] == image_id,
            coco_annotation
        )
    )
    annotations = list(
        map(
            lambda x: {
                'category_id': x['category_id'],
                'bbox': x['bbox']
            },
            annotations
        )
    )

    image = cv2.imread(image_path + image_file_name)

    for annotation in annotations:

        category_id = annotation['category_id']
        bbox = annotation['bbox']

        x, y, w, h = bbox

        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (0, 255, 0),
            2
        )

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print(annotations)

    plt.show()

# %%


def filter_and_convert_coco_to_yolo(coco_json_path, image_dir, output_image_dir, output_label_dir):
    """
    Filter COCO dataset and convert annotations to YOLO format while copying images.

    :param coco_json_path: Path to the COCO JSON file.
    :param image_dir: Path to the COCO images directory.
    :param output_image_dir: Directory to save filtered images.
    :param output_label_dir: Directory to save YOLO annotations.
    """
    # Load COCO JSON annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = {image["id"]: image for image in coco_data["images"]}
    annotations = coco_data["annotations"]

    category_counts = {0: 0, 1: 0, "other": 0}  # For category limits
    yolo_annotations = {}

    for ann in annotations:
        image_id = ann["image_id"]
        category_id = ann["category_id"] - 1  # YOLO expects 0-based indexing for class IDs
        bbox = ann["bbox"]

        # Filter by category limits
        if category_id == 0 and category_counts[0] >= MAX_IMAGES_NO_LABEL:
            continue
        elif category_id == 1 and category_counts[1] >= MAX_IMAGES_PERSON:
            continue
        elif category_id != 0 and category_id != 1 and category_counts["other"] >= MAX_IMAGES_REST_OF_CATEGORIES:
            continue

        # Increment the count for the category
        if category_id == 0:
            category_counts[0] += 1
        elif category_id == 1:
            category_counts[1] += 1
        else:
            category_counts["other"] += 1

        # Get image dimensions
        image_info = images[image_id]
        image_width = image_info["width"]
        image_height = image_info["height"]

        # Convert bounding box to YOLO format
        yolo_bbox = convert_bbox_coco_to_yolo(image_width, image_height, bbox)

        # Add to annotations for this image
        if image_id not in yolo_annotations:
            yolo_annotations[image_id] = []
        yolo_annotations[image_id].append([category_id, *yolo_bbox])

    # Write YOLO annotations and move images
    for image_id, annotations in yolo_annotations.items():
        image_info = images[image_id]
        image_filename = image_info["file_name"]
        output_label_path = os.path.join(output_label_dir, f"{os.path.splitext(image_filename)[0]}.txt")
        output_image_path = os.path.join(output_image_dir, image_filename)

        # Copy the image to the new directory
        original_image_path = os.path.join(image_dir, image_filename)
        if os.path.exists(original_image_path):
            cv2.imwrite(output_image_path, cv2.imread(original_image_path))

        # Write YOLO annotations to text file
        with open(output_label_path, 'w') as f:
            for ann in annotations:
                f.write(" ".join(map(str, ann)) + "\n")

    # print all categories counts
    print(category_counts)


# %%
# Wrap the main execution logic in a function or conditional block
# This prevents it from running automatically if the script is imported
if __name__ == "__main__":
    print(f"Using Image Path: {image_path}")
    print(f"Using Annotation File: {annotation_file}")
    print(f"Output Images Path: {output_images}")
    print(f"Output Labels Path: {output_labels}")

    filter_and_convert_coco_to_yolo(annotation_file, image_path, output_images, output_labels)
    print("Processing complete.")

# # %%
# labels = os.listdir(output_labels)
# no_label = list(filter(lambda x: not x.endswith("txt"), labels))

