# Augmented Reality Application for Maintenance Knowledge Management through Interactive Visualization

<center>
    Jose Daniel Sarmiento , Manuel Ayala  | { jose2192232, jose2195529 } @correo.uis.edu.co
</center>






This repository contains scripts for processing the COCO dataset, specifically for filtering images based on category limits and converting annotations from COCO format to YOLO format, and for augmenting the resulting dataset.

## Pipeline 1: Filtering and COCO to YOLO Conversion

The `pipeline_1.py` script filters images from the COCO dataset based on specified category limits and converts their annotations to the YOLO format. Filtered images are saved to a specified output directory, and corresponding YOLO annotation files (.txt) are saved to another specified directory.

### Prerequisites

-   Python 3.x

```sh
conda create --name train_data  python=3.10 -y
conda activate train_data

pip install -r requirements.txt
```

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

## Pipeline 3: Image Quantization and Filtering

The `pipeline_3.py` script processes a dataset that has been split into `train`, `test`, and `val` sets. It filters images based on the presence of a specific class label (hardcoded as class ID 80 in the script), resizes the filtered images to a fixed height (640px) while maintaining aspect ratio, reduces their JPEG quality, and saves them along with their corresponding label files to a structured output directory.

### Prerequisites

-   Python 3.x
-   Required Python libraries: `numpy`, `pandas`, `opencv-python`, `matplotlib`, `plotly` (Install using `pip install numpy pandas opencv-python matplotlib plotly`)
-   A dataset split into `train`, `test`, and `val` subdirectories, each containing `images` and `labels` subdirectories (e.g., output from a dataset splitting tool). The labels should be in YOLO format.

### Running the Script

Execute the script from the command line, providing the paths to the split dataset and the desired output directory:

```bash
python pipeline_3.py --input_dir /path/to/split_dataset \
                     --output_dir /path/to/quantized_output
```

### Arguments

-   `--input_dir`: (Optional) Path to the base directory containing the split dataset (`train`, `test`, `val` folders). Defaults to `./split_dataset`.
-   `--output_dir`: (Optional) Directory where the quantized and filtered output will be saved. Defaults to `./quantize/`. The script will create `train`, `test`, and `val` subdirectories within this output directory.

### Example

If your split dataset is in `./split_data` and you want to save the quantized output to `./quantized_data`:

```bash
python pipeline_3.py --input_dir ./split_data \
                     --output_dir ./quantized_data
```

The script will process the `train`, `test`, and `val` sets, filter images containing class ID 80, resize and compress them, and save the results in `./quantized_data/train/images`, `./quantized_data/train/labels`, etc. It will print the input and output paths for each image being processed.

## Pipeline 4: Model Training

The `train.sh` script provides a convenient way to start YOLOv8 detection model training using specified parameters or defaults. It simplifies the command-line execution of the `yolo` training task.

### Prerequisites

-   [Ultralytics YOLOv8](https://docs.ultralytics.com/) installed.
-   A prepared dataset with a corresponding `data.yml` file (e.g., the output from previous pipelines).
-   (Optional) `wandb` account and logged in if you want to use Weights & Biases for tracking.

### Running the Script

Make the script executable first:

```bash
chmod +x train.sh
```

Execute the script from the command line, optionally providing arguments in the specified order:

```bash
./train.sh [model] [data_yaml] [epochs] [imgsz] [device] [batch]
```

### Arguments (Positional)

1.  `model`: (Optional) The YOLO model to use for training (e.g., `yolov8n.pt`, `yolov8s.pt`). Defaults to `yolov8n.pt`.
2.  `data_yaml`: (Optional) Path to the dataset configuration file (`.yml`). Defaults to `/content/data.yml`.
3.  `epochs`: (Optional) Number of training epochs. Defaults to `100`.
4.  `imgsz`: (Optional) Input image size for training. Defaults to `640`.
5.  `device`: (Optional) Device to run training on (e.g., `0` for GPU 0, `cpu`). Defaults to `0`.
6.  `batch`: (Optional) Batch size for training. `-1` for auto-batch. Defaults to `-1`.

### Example

To train a `yolov8s.pt` model using the dataset defined in `./quantized_data/data.yaml` for 50 epochs with an image size of 640 on GPU 0 and auto-batch size:

```bash
./train.sh yolov8s.pt ./quantized_data/data.yaml 50 640 0 -1
```

To run with all default values:

```bash
./train.sh
```

The script will print the parameters being used before starting the training process.

## Pipeline 5: Model Export

The `export.sh` script facilitates exporting a trained YOLOv8 model (e.g., from Pipeline 4) to the TFLite format, which is suitable for deployment on mobile devices. It includes Non-Maximum Suppression (NMS) in the exported graph and sets a default image size.

### Prerequisites

-   [Ultralytics YOLOv8](https://docs.ultralytics.com/) installed.
-   A trained YOLOv8 model file (e.g., `best.pt` from the training output).

### Running the Script

Make the script executable first:

```bash
chmod +x export.sh
```

Execute the script from the command line, optionally providing the path to the model file:

```bash
./export.sh [model_path]
```

### Arguments (Positional)

1.  `model_path`: (Optional) Path to the trained YOLOv8 model file (`.pt`). Defaults to `best.pt` in the current directory.

### Example

To export the default `best.pt` model located in the current directory:

```bash
./export.sh
```

To export a specific model, for example, located in the training results directory:

```bash
./export.sh runs/detect/train/weights/best.pt
```

The script will run the `yolo export` command with the specified (or default) model, setting the format to `tflite`, enabling `nms`, and using an image size of `640`. The exported TFLite model will typically be saved in the same directory as the input `.pt` file with a `_saved_model` suffix and then further converted within that subdirectory.

