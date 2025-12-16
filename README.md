# Computer Vision - Waste Detection

A real-time waste detection system using YOLOv8 segmentation to classify objects as recycling or trash.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install all necessary packages including `ultralytics`, `opencv-python`, `torch`, and other dependencies needed for training and inference.

## Python Scripts

### `data_prep.py`
Prepares the TACO dataset for training by:
- Converting COCO format annotations to YOLOv8 format
- Creating a stratified 80/10/10 train/val/test split
- Mapping 60 TACO categories to binary classes (recycling/trash)
- Generating the `data.yaml` configuration file

**Setup:**
Before running this script, you need to download the TACO dataset:
1. Clone the TACO repository: `git clone https://github.com/pedropro/TACO`
2. Follow the instructions in the [TACO repository](https://github.com/pedropro/TACO) to download the dataset images
3. Ensure the `data` folder with `annotations.json` is in the project root

**Usage:**
```bash
python data_prep.py
```

**Tips:**
- Uses a fixed random seed (42) for reproducible train/val/test splits
- The stratified split ensures balanced distribution of recycling and trash classes across all splits
- Creates the required directory structure automatically (`data/images/` and `data/labels/` for train/val/test)
- Progress is displayed with a progress bar showing counts for each split
- After completion, the `data.yaml` file is generated for YOLOv8 training

### `train.py`
Trains a YOLOv8 segmentation model on the prepared dataset.

**Usage:**
```bash
python train.py
```

**Features:**
- Uses `yolov8m-seg.pt` as base model (medium segmentation model)
- Trains for 100 epochs
- Image size: 960x960
- Batch size: 8 (adjust based on GPU memory)
- Saves checkpoints every 10 epochs (`save_period=10`)
- Best weights saved to `runs/segment/taco_seg_binary/weights/best.pt`

**Tips:**
- Training time varies significantly based on your GPU (can take several hours)
- Reduce `batch` size if you run out of GPU memory (e.g., `batch=4` or `batch=2`)
- Lower `imgsz` (e.g., 640 or 416) for faster training with less memory usage
- The script will automatically download `yolov8m-seg.pt` on first run
- After training completes, copy `best.pt` from the runs directory to the project root for inference scripts
- Training progress, metrics, and visualizations are saved in the `runs/segment/taco_seg_binary/` directory

### `main.py`
Optimized real-time waste detection using webcam feed.

**Usage:**
```bash
python main.py
```

**Features:**
- GPU acceleration (MPS on Mac, CUDA on NVIDIA GPUs)
- Low latency optimizations (reduced image size, frame skipping support)
- Error handling and camera fallback options
- Press 'q' to quit

**Tips:**
- Automatically detects and uses the best available device (MPS > CUDA > CPU)
- Camera resolution is set to 640x480 for faster processing
- Inference is run at 416x416 resolution (faster than default 640)
- Uses half precision (FP16) on GPU for ~2x speedup
- The `frame_skip` variable (line 57) controls frame processing rate:
  - `frame_skip=1`: Process every frame (best accuracy, slower)
  - `frame_skip=2`: Process every other frame (2x faster)
  - `frame_skip=3`: Process every third frame (3x faster)
- For even lower latency, reduce `imgsz` parameter (line 83) to 320
- On Mac, ensure camera permissions are granted in System Settings > Privacy & Security > Camera
- The script tries multiple camera backends and indices if the default fails
- Window shows camera feed immediately, even before first detection completes

### `main_simple.py`
Simple version of real-time waste detection without optimizations.

**Usage:**
```bash
python main_simple.py
```

**Features:**
- Basic webcam feed with waste detection
- No performance optimizations
- Good for testing or when you don't need low latency
- Press 'q' to quit

**Tips:**
- Processes every frame at full resolution (slower but more accurate)
- Simpler code structure, easier to understand and modify
- Good for initial testing and debugging
- Runs inference on every single frame (no frame skipping)
- May have lower FPS compared to `main.py` due to lack of optimizations
- Uses default confidence threshold of 0.4

## Model

The trained model should be saved as `best.pt` in the project root for the inference scripts to work.

**Getting the model:**
- After training with `train.py`, copy the best weights: 
  ```bash
  cp runs/segment/taco_seg_binary/weights/best.pt ./best.pt
  ```
- Or use your own trained YOLOv8 segmentation model in `.pt` format

## Workflow

1. **Prepare data**: Run `data_prep.py` after downloading the TACO dataset
2. **Train model**: Run `train.py` to train on the prepared data
3. **Copy weights**: Copy `best.pt` to project root after training
4. **Run inference**: Use `main.py` (optimized) or `main_simple.py` (simple) for real-time detection

