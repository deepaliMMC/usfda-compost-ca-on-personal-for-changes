# livestockkk

> Estimates the number of grazing animals on California pasturelands.  
> Provides FDA with risk assessment data based on livestock density near produce farms.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project detects animals from 3-band aerial imagery (RGB) using YOLOv11 object detection:

**Data**: 3-band aerial imagery (RGB) from AWS S3

**Model**: YOLOv11n trained on 512×512 image chips

**Approach**: Single-class object detection with positive/negative sampling and spatial split for validation

## 📊 Executive Summary

### Mission

Estimate and localize grazing animals (cattle) across California pasturelands using aerial imagery.  
Outputs support **FDA food safety risk assessment** by quantifying livestock density near produce farms.

### Technical Approach

**Input Data:** 3-band aerial imagery (RGB) from AWS S3, chip-based sampling from labeled polygons and negative point samples.

**Model Architecture:**
- **Primary:** YOLOv11 (tiny variant) optimized for small objects
- **Input Size:** 512×512 pixels (3 channels)
- **Training Strategy:** Single-class object detection with bounding box regression
- **Data Split:** Spatial tile-based train/val split to prevent data leakage

**Output Artifacts:**
- Per-chip predictions with bounding boxes and confidence scores
- Training curves (loss, mAP, precision, recall)
- Model checkpoints and dataset configuration
- Automated dataset YAML for YOLO format

### Performance Metrics

| Metric | Target | Achieved | Interpretation |
|--------|--------|----------|----------------|
| mAP@0.5 | ≥ 0.75 |  **0.98**  | Mean Average Precision at IoU 0.5 |
| mAP@0.5:0.95 | ≥ 0.60 | **0.45** | Mean Average Precision across IoU thresholds |
| Precision | ≥ 0.80 | **0.8355** | Minimize false positives |
| Recall | ≥ 0.75 | **0.8186**  | Minimize false negatives |

**Operational Interpretation:** YOLOv11n provides the best balance between accuracy and speed for animal detection. The model handles smaller animal sizes and orientations while maintaining robust performance across different imagery tiles and years.

### Release Quality Gates

| Category | Metric | Target | Purpose |
|----------|--------|--------|---------|
| **Core Quality** | mAP@0.5 (val) | ≥ 0.75 | Overall detection effectiveness |
| | mAP@0.5:0.95 (val) | ≥ 0.60 | Localization accuracy |
| | Precision (val) | ≥ 0.80 | False positive control |
| | Recall (val) | ≥ 0.75 | False negative control |
| **Localization** | IoU threshold | ≥ 0.50 | Bounding box accuracy |
| | Box loss convergence | < 0.05 | Stable localization |
| **Robustness** | Cross-tile mAP drop | ≤ 10% | Generalization across regions |
| | Year-to-year consistency | ≥ 0.90 | Temporal stability |
| **Ops** | Inference speed (FP16) | ≥ 100 img/s/GPU | Meets production SLAs |
| | GPU memory usage | ≤ 16GB | Deployment efficiency |
| **Data QA** | Valid chips generated | ≥ 99% | Clean training pipeline |
| | Label coverage | 100% | All chips properly labeled |

## Project Structure

```
compost-detect/
├── configs/
│   └── default.yaml               # Main configuration settings
├── data/
│   └── .gitkeep                   # Placeholder for data directory
├── scripts/
│   ├── setup_env.sh               # Environment setup script
│   └── sync_s3_data.sh            # S3 data synchronization
├── src/
│   └── compost-detect/
│       ├── __init__.py
│       ├── data_generation.py     # Generate training chips from NAIP
│       ├── training.py            # YOLOv8 training pipeline
│       ├── prediction.py          # Inference on new imagery
│       ├── post_processing.py     # Results analysis and conversion
│       ├── data/
│       │   └── __init__.py
│       ├── models/
│       │   └── __init__.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py          # Configuration management
│           ├── geo_utils.py       # Geospatial utilities
│           ├── io_utils.py        # I/O helpers (TIFF, YOLO format)
│           ├── logging_utils.py   # Logging setup
│           └── s3_utils.py        # S3 operations
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest configuration
│   ├── test_data_generation.py   # Data generation tests
│   ├── test_models.py             # Model tests
│   └── test_utils.py              # Utility function tests
├── LICENSE
├── pyproject.toml                 # Project metadata and dependencies
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd livestock-est-ca
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- `ultralytics>=8.0.0` - YOLO framework
- `rasterio` - Geospatial raster I/O
- `geopandas` - Vector data handling
- `shapely` - Geometric operations
- `boto3` - AWS S3 integration
- `numpy` - Numerical operations
- `tqdm` - Progress bars

### 4. Configure AWS credentials

```bash
aws configure
# Enter your AWS Access Key ID, Secret Key, and region
```

### 5. Setup environment variables

Create a `.env` file or export environment variables:

```bash
export AWS_DEFAULT_REGION=us-east-1
export MAX_WORKERS=24  # Adjust based on your system
```

### 6. Verify GDAL/Rasterio

```bash
python -c "import rasterio; print(rasterio.__version__)"
```

### 7. Run setup script

```bash
bash scripts/setup_env.sh
```

## Usage

### Configuration

Edit `configs/default.yaml` to customize:

```yaml
# S3 Configuration
BUCKET: "cafo-naip-data"
RGB_PREFIX: "processed/"
OUT_PREFIX: "basemap18"

# Input Files
POS_GEOJSON: "positive_polygons.geojson"
NEG_GEOJSON: "neg_points.geojson"

# Data Generation
CHIP_SIZE: 512
NEG_BUFFER_M: 200.0
POS_CLASS_ID: 0
MAX_WORKERS: 24

# Training Parameters
IMG_SIZE: 512
EPOCHS: 200
BATCH: 64
LR0: 0.0005
WEIGHT_DECAY: 0.0005
NUM_WORKERS: 8
SAVE_PERIOD: 10

# Classes
CLASS_NAMES: ["animal"]
```

Or override via environment variables.

### 1. Data Generation

Create training chips from NAIP imagery and labeled polygons/points.

```bash
python src/animal-detect/data_generation.py
```

**What it does:**
- Fetches RGB tile list from S3
- Reads positive polygon labels (animals)
- Reads negative point samples (non-animals)
- Generates 512*512 chips centered on facilities
- Creates negative samples with buffer distance
- Converts polygons to YOLO format bounding boxes
- Splits data into train/val based on tile location
- Uploads chips and labels to S3

**Inputs:**
- 3-band COGs on S3 (RGB)
- `compost_polygons.geojson` - Polygon geometries of individual animals
- `neg_points.geojson` - Point locations for negative samples

**Outputs:**
- `s3://{BUCKET}/{OUT_PREFIX}/images/train/*.tif` - Training chips
- `s3://{BUCKET}/{OUT_PREFIX}/images/val/*.tif` - Validation chips
- `s3://{BUCKET}/{OUT_PREFIX}/labels/train/*.txt` - YOLO format labels
- `s3://{BUCKET}/{OUT_PREFIX}/labels/val/*.txt` - YOLO format labels
- `s3://{BUCKET}/{OUT_PREFIX}/dataset.yaml` - Dataset configuration

**Key Features:**
- **Resume-safe**: Skips already processed tiles
- **Spatial split**: Train/val split based on tile location (prevents data leakage)
- **Negative sampling**: Maintains distance buffer from positive samples
- **Parallel processing**: Multi-worker tile processing
- **Quality checks**: Validates chip reads and geometry intersections

### 2. Training

Train YOLOv11n model on generated dataset.

```bash
python src/animal-detect/training.py
```

**Training Configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | YOLOv11n | Medium variant (25M parameters) |
| Image Size | 512×512 | Input resolution |
| Batch Size | 12 | Fits 24GB GPU |
| Epochs | 200 | Maximum training epochs |
| Learning Rate | 5e-4 | Initial LR (with warmup) |
| Weight Decay | 5e-4 | L2 regularization |
| Workers | 16 | DataLoader processes |
| Save Period | 10 | Checkpoint every N epochs |

**Features:**
- 3-channel input support (RGB)
- Automatic mixed precision (AMP)
- Cosine learning rate scheduling with warmup
- Data augmentation (mosaic, flip, scale, crop)
- Multi-GPU support (DDP)
- Checkpoint saving (best + periodic)
- TensorBoard logging
- Validation during training

**Outputs:**
- `{PROJECT_DIR}/{RUN_NAME}/weights/best.pt` - Best model checkpoint
- `{PROJECT_DIR}/{RUN_NAME}/weights/last.pt` - Latest checkpoint
- `{PROJECT_DIR}/{RUN_NAME}/results.csv` - Training metrics
- `{PROJECT_DIR}/{RUN_NAME}/results.png` - Training curves
- `{PROJECT_DIR}/{RUN_NAME}/confusion_matrix.png` - Confusion matrix

**Monitoring Training:**

```bash
# View training logs
tail -f {PROJECT_DIR}/{RUN_NAME}/train.log

# TensorBoard (if enabled)
tensorboard --logdir {PROJECT_DIR}/{RUN_NAME}

# Check results CSV
cat {PROJECT_DIR}/{RUN_NAME}/results.csv
```

### 3. Inference

Run predictions on new RGB imagery.

```bash
python src/animal-detect/prediction.py
```

**What it does:**
- Loads trained YOLOv11 model
- Reads 3-channel test images
- Runs inference with configurable thresholds
- Saves bounding boxes predicte on dannotated images as geojson
- Outputs prediction results

**Outputs:**
- Annotated images with detections
- Bounding box coordinates and confidence scores as geojson
- JSON/CSV with detection results

**Configuration:**

```python
# In prediction.py or via settings
model = YOLO(settings.YOLO_MODEL_PATH)

results = model.predict(
    source=settings.IMAGE_PATH,
    imgsz=512,
    conf=0.25,      # Confidence threshold
    iou=0.5,        # NMS IoU threshold
    save=True,      # Save annotated images
    show=False      # Display results
)
```

### 4. Post-Processing

Convert TIFF chips to PNG and perform analysis.

```bash
python src/compost-detect/post_processing.py
```

**What it does:**
- Converts 4-band TIFF chips to PNG format
- Preserves all bands in conversion
- Backs up original TIFFs
- Prepares data for visualization/analysis

**Outputs:**
- PNG images in each split directory
- Original TIFFs backed up to `tiff_backup/` folders

## Configuration Details

### Key Parameters

**Data Generation:**
- `CHIP_SIZE`: 1024px square chips (standard for YOLOv8 at high res)
- `NEG_BUFFER_M`: 200m buffer around positive samples for negative mining
- `MAX_WORKERS`: Parallel workers for tile processing (recommend 24-32 on high-core machines)

**Training:**
- `EPOCHS`: 100 (early stopping based on validation mAP)
- `BATCH`: 12 (adjust based on GPU memory)
- `LR0`: 5e-4 (initial learning rate with warmup)
- `WEIGHT_DECAY`: 5e-4 (L2 regularization)
- `IMG_SIZE`: 1024 (maintain consistency with chip size)

**Model:**
- YOLOv8m: Best balance of speed and accuracy
- 4-channel input: RGB + NIR for enhanced detection
- Single class: "compost"

### Data Split Strategy

**Spatial Split (Tile-Based):**
- Training and validation sets split by NAIP tile location
- Prevents spatial autocorrelation and data leakage
- Validation tiles are geographically separated from training tiles
- Split determined by `tile_split_is_val()` function in `geo_utils.py`

**Advantages:**
- More realistic performance estimates
- Tests model generalization to new geographic areas
- Avoids overfitting to specific tile characteristics

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 1× 16GB (e.g., Tesla T4) | 1× 24GB (e.g., RTX 3090, A10G) |
| **System RAM** | 32GB | 64GB+ |
| **Storage** | 200GB SSD | 500GB NVMe SSD |
| **CPU** | 8 cores | 16+ cores (for data generation) |

**Tested Configurations:**
- AWS `ml.g5.xlarge` (1× A10G 24GB) - optimal for training
- AWS `ml.g5.2xlarge` (1× A10G 24GB) - faster training with more CPU
- AWS `ml.p3.2xlarge` (1× V100 16GB) - requires batch size reduction

## Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size in config
BATCH: 8  # or lower

# Reduce image size (trades accuracy)
IMG_SIZE: 640

# Enable gradient accumulation in YOLOv8
# (set in ultralytics training arguments if needed)
```

### S3 Access Issues

```bash
# Verify credentials
aws s3 ls s3://{BUCKET}/

# Check bucket permissions
aws s3api get-bucket-policy --bucket {BUCKET}

# Test file access
aws s3 cp s3://{BUCKET}/{NAIP_PREFIX}/sample.tif ./test.tif

# Verify GDAL VSI access
gdalinfo /vsis3/{BUCKET}/{NAIP_PREFIX}/sample.tif
```

### GDAL/Rasterio Issues

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev

# macOS
brew install gdal

# Verify installation
gdalinfo --version
python -c "import rasterio; print(rasterio.__version__)"

# Reinstall rasterio if needed
pip uninstall rasterio
pip install --no-cache-dir rasterio
```

Contact: info@mapmycrop.com
Version: v2.0 – October 2025
