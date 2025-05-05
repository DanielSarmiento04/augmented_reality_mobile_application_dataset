# augmented_reality_mobile_application-_dataset

This repository contains scripts for processing the COCO dataset, specifically for filtering images based on category limits and converting annotations from COCO format to YOLO format, and for augmenting the resulting dataset.

## Pipeline 1: Filtering and COCO to YOLO Conversion

The `pipeline_1.py` script filters images from the COCO dataset based on specified category limits and converts their annotations to the YOLO format. Filtered images are saved to a specified output directory, and corresponding YOLO annotation files (.txt) are saved to another specified directory.

### Prerequisites

-   Python 3.x
-   Required Python libraries: `numpy`, `pandas`, `opencv-python`, `matplotlib` (Install using `pip install numpy pandas opencv-python matplotlib`)
-   COCO dataset images and annotation file (`instances_train2017.json` or similar).

### Running the Script

Execute the script from the command line, providing the necessary paths as arguments:

```bash
python pipeline_1.py --image_path /path/to/coco/images \
                    --annotation_path /path/to/coco/annotations \
                    --output_images ./filtered_images \
                    --output_labels ./filtered_labels
```

### Arguments

-   `--image_path`: (Required) Path to the directory containing the COCO image files (e.g., `train2017/`).
-   `--annotation_path`: (Required) Path to the directory containing the COCO annotation JSON file (e.g., `annotations/`). The script specifically looks for `instances_train2017.json` within this directory.
-   `--output_images`: (Optional) Directory where the filtered images will be saved. Defaults to `./images/`.
-   `--output_labels`: (Optional) Directory where the generated YOLO annotation files (.txt) will be saved. Defaults to `./labels/`.

### Example

If your COCO images are in `/data/coco/train2017` and annotations are in `/data/coco/annotations`, and you want to save the filtered output to `./output/images` and `./output/labels`:

```bash
python pipeline_1.py --image_path /data/coco/train2017 \
                    --annotation_path /data/coco/annotations \
                    --output_images ./output/images \
                    --output_labels ./output/labels
```

The script will print the paths being used and indicate when processing is complete. It will also print the final counts for each category group after filtering.

## Pipeline 2: Data Augmentation

The `pipeline_2.py` script takes a dataset of images and corresponding YOLO-formatted labels (like the output from `pipeline_1.py`) and applies various augmentations to increase the dataset size and diversity. It saves the augmented images and their labels to specified output directories. The original images and labels are also copied to the output directories.

### Prerequisites

-   Python 3.x
-   Required Python libraries: `opencv-python`, `albumentations`, `numpy`, `tqdm` (Install using `pip install opencv-python albumentations numpy tqdm`)
-   A dataset consisting of images and corresponding YOLO annotation files (.txt).
-   A `classes.txt` file listing the class names, one per line (optional, defaults to `./classes.txt`).

### Running the Script

Execute the script from the command line:

```bash
python pipeline_2.py --input_images_dir /path/to/input/images \
                     --input_labels_dir /path/to/input/labels \
                     --output_images_dir /path/to/output/augmented_images \
                     --output_labels_dir /path/to/output/augmented_labels \
                     --classes_file /path/to/classes.txt \
                     --num_augmentations 10
```

### Arguments

-   `--input_images_dir`: (Required) Path to the directory containing the input images.
-   `--input_labels_dir`: (Required) Path to the directory containing the input YOLO label files (.txt).
-   `--output_images_dir`: (Required) Directory where the augmented images will be saved.
-   `--output_labels_dir`: (Required) Directory where the augmented YOLO label files (.txt) will be saved.
-   `--classes_file`: (Optional) Path to the file containing class names. Defaults to `./classes.txt`.
-   `--num_augmentations`: (Optional) Number of augmented versions to generate for each input image. Defaults to `20`.

### Example

If your filtered images from Pipeline 1 are in `./output/images` and labels are in `./output/labels`, and you want to save the augmented dataset to `./augmented/images` and `./augmented/labels`, generating 15 augmentations per image:

```bash
python pipeline_2.py --input_images_dir ./output/images \
                     --input_labels_dir ./output/labels \
                     --output_images_dir ./augmented/images \
                     --output_labels_dir ./augmented/labels \
                     --num_augmentations 15
```

The script will process each image, apply the specified number of augmentations, save the results, and finally copy the original files to the output directories. Progress is shown using a progress bar.

