# Trash Sorting Education

An educational project combining computer vision waste detection with an interactive quiz system. The project includes YOLOv8-based waste classification (recycling vs. trash) and a MediaPipe-powered quiz application for waste sorting education.

## Project Structure

```
TrashSortingEdu/
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LiveDetection/            # Real-time waste detection scripts
│   ├── main.py              # Optimized real-time waste detection
│   ├── main_simple.py       # Simple waste detection (no optimizations)
│   ├── best.pt              # Trained YOLOv8 model weights
│   └── best_old.pt          # Previous model version
└── YoloTraining/            # Model training and data preparation
│    ├── data_prep.py         # TACO dataset preparation with augmentation
│    ├── data_prep_old.py     # Previous version (no augmentation)
│    ├── train.py             # YOLOv8 segmentation model training
│    └── test_model.py        # Model evaluation on test set
└── GestureDetection/         # interactive quiz with gesture detection
    ├── fingerRaise.py        # finger raise quiz
    ├── pinchAndSort.py       # pinch and sort quiz
    ├── data
    └── utils
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install all necessary packages including `ultralytics`, `opencv-python`, `torch`, `mediapipe`, and other dependencies needed for training, inference, and the quiz application.

## Scripts Overview

### GestureDetection/ Directory - Interactive Trash Sorting Quiz

An educational application that has two interactive modes: a multiple-choice quiz where users answer questions by holding up 1–4 fingers, and a gesture-based sorting game where users pinch and drag virtual trash items into the correct bins.

**Features:**

- Real-time hand gesture recognition (1-4 fingers = A, B, C, D choices)
- Multiple-choice questions about waste sorting
- Gesture-based pinch interaction to grab and sort virtual trash items into bins
- Score tracking and progress visualization
- Dwell-time confirmation to prevent accidental selections
- Visual feedback with color-coded responses

**Usage:**

```bash
cd GestureDetection
python fingerRaise.py
python pinchAndSort.py
```

**Controls:**

Finger Raise MCQ:
- Hold up 1-4 fingers to select answer (1=A, 2=B, 3=C, 4=D)
- Hold steady for 2 seconds to confirm selection
- Press 'q' to quit
- Press 'n' to skip to next question

Pinch and Sort:		
- Pinch your thumb and index finger together to grab a trash item
- Drag the item by moving your pinched fingers
- Release the pinch to drop the item into a bin
- Drop items into the correct bin (Recycling, Compost, or Landfill) to score points
- Sort as many items as possible before the timer runs out
- Press 'q' to quit
- 
**Tips:**

- Ensure good lighting for hand detection
- Keep hand clearly visible in camera frame
- Camera feed is mirrored for natural interaction
- Quiz automatically progresses through all questions

### LiveDetection/ Directory

#### `LiveDetection/main.py` - Optimized Real-Time Waste Detection

High-performance real-time waste detection using webcam feed with GPU acceleration and optimization features.

**Features:**

- GPU acceleration (MPS on Mac, CUDA on NVIDIA GPUs)
- Low latency optimizations (reduced image size, frame skipping support)
- Error handling and camera fallback options
- Automatic device detection (MPS > CUDA > CPU)
- Half precision (FP16) inference for faster processing

**Usage:**

```bash
cd LiveDetection
python main.py
```

**Tips:**

- Automatically detects and uses the best available device
- Camera resolution is set to 640x480 for faster processing
- Inference runs at 416x416 resolution (faster than default 640)
- The `frame_skip` variable (line 57) controls frame processing rate:
  - `frame_skip=1`: Process every frame (best accuracy, slower)
  - `frame_skip=2`: Process every other frame (2x faster)
  - `frame_skip=3`: Process every third frame (3x faster)
- For even lower latency, reduce `imgsz` parameter (line 83) to 320
- On Mac, ensure camera permissions are granted in System Settings > Privacy & Security > Camera
- Press 'q' to quit

#### `LiveDetection/main_simple.py` - Simple Waste Detection

Basic version of real-time waste detection without performance optimizations, ideal for testing and debugging.

**Usage:**

```bash
cd LiveDetection
python main_simple.py
```

**Features:**

- Basic webcam feed with waste detection
- No performance optimizations
- Simpler code structure, easier to understand and modify
- Processes every frame at full resolution
- Uses default confidence threshold of 0.4

**Tips:**

- Good for initial testing and debugging
- May have lower FPS compared to `main.py` due to lack of optimizations
- Press 'q' to quit

### YoloTraining/ Directory

#### `YoloTraining/data_prep.py` - Dataset Preparation with Augmentation

Prepares the TACO dataset for training by converting COCO format annotations to YOLOv8 format with instance-safe data augmentation.

**Features:**

- Converts COCO format annotations to YOLOv8 segmentation format
- Creates stratified 80/10/10 train/val/test split
- Maps 60 TACO categories to binary classes (recycling/trash)
- Applies instance-safe augmentation (2x per image = 3x dataset size)
- Generates `data.yaml` configuration file

**Setup:**
Before running this script, you need to download the TACO dataset:

1. Clone the TACO repository: `git clone https://github.com/pedropro/TACO`
2. Follow the instructions in the [TACO repository](https://github.com/pedropro/TACO) to download the dataset images
3. Ensure the `data` folder with `annotations.json` is in the project root

**Usage:**

```bash
cd YoloTraining
python data_prep.py
```

**Tips:**

- Uses a fixed random seed (42) for reproducible train/val/test splits
- Stratified split ensures balanced distribution of recycling and trash classes
- Creates required directory structure automatically (`data/images/` and `data/labels/` for train/val/test)
- Augmentation includes: horizontal flip, brightness/contrast, affine transforms, Gaussian blur
- Progress is displayed with progress bars
- After completion, the `data.yaml` file is generated for YOLOv8 training

#### `YoloTraining/train.py` - Model Training

Trains a YOLOv8 segmentation model on the prepared dataset.

**Usage:**

```bash
cd YoloTraining
python train.py
```

**Features:**

- Uses `yolov8m-seg.pt` as base model (medium segmentation model)
- Trains for 50 epochs
- Image size: 960x960
- Batch size: 8 (adjust based on GPU memory)
- Saves checkpoints every 10 epochs (`save_period=10`)
- Best weights saved to `runs/segment/taco_seg_binary/weights/best.pt`

**Tips:**

- Training time varies significantly based on your GPU (can take several hours)
- Reduce `batch` size if you run out of GPU memory (e.g., `batch=4` or `batch=2`)
- Lower `imgsz` (e.g., 640 or 416) for faster training with less memory usage
- The script will automatically download `yolov8m-seg.pt` on first run
- After training completes, copy `best.pt` to `LiveDetection/` directory for inference scripts
- Training progress, metrics, and visualizations are saved in the `runs/segment/taco_seg_binary/` directory

#### `YoloTraining/test_model.py` - Model Evaluation

Evaluates the trained model on the test set and generates metrics and prediction visualizations.

**Usage:**

```bash
cd YoloTraining
python test_model.py
```

**Features:**

- Evaluates model on test split
- Computes box and mask mAP metrics (mAP50 and mAP50-95)
- Saves prediction images and annotations
- Generates performance plots
- Copies sample predictions to `test_predictions/` directory

**Tips:**

- Update `BEST_MODEL` path (line 5) if experiment number changed
- Results saved to `runs/segment/val/`
- Sample predictions copied to `test_predictions/` directory (20 images by default)
- Check metrics to assess model performance before deployment

#### `YoloTraining/data_prep_old.py` - Legacy Data Preparation

Previous version of data preparation script without augmentation. Kept for reference but `data_prep.py` is recommended.

## Model Files

The trained model should be saved as `best.pt` in the `LiveDetection/` directory for the inference scripts to work.

**Getting the model:**

- After training with `train.py`, copy the best weights:
  ```bash
  cp runs/segment/taco_seg_binary/weights/best.pt LiveDetection/best.pt
  ```
- Or use your own trained YOLOv8 segmentation model in `.pt` format

## Complete Workflow

1. **Prepare data**:

   ```bash
   cd YoloTraining
   python data_prep.py
   ```

   (After downloading the TACO dataset)

2. **Train model**:

   ```bash
   cd YoloTraining
   python train.py
   ```

3. **Evaluate model** (optional):

   ```bash
   cd YoloTraining
   python test_model.py
   ```

4. **Copy weights**:

   ```bash
   cp runs/segment/taco_seg_binary/weights/best.pt LiveDetection/best.pt
   ```

5. **Run inference**:

   ```bash
   cd LiveDetection
   python main.py          # Optimized version
   # or
   python main_simple.py   # Simple version
   ```

6. **Run quiz application**:
   ```bash
   python main.py          # From project root
   ```
