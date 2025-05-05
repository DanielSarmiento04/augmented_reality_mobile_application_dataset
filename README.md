# augmented_reality_mobile_application-_dataset

This repository contains scripts for processing the COCO dataset, specifically for filtering images based on category limits and converting annotations from COCO format to YOLO format.

## Usage

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
python pipline_1.py --image_path /data/coco/train2017 \
                    --annotation_path /data/coco/annotations \
                    --output_images ./output/images \
                    --output_labels ./output/labels
```

The script will print the paths being used and indicate when processing is complete. It will also print the final counts for each category group after filtering.

