# livestock

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
### 4. Post-Processing and Spatial Quality Validation

Following model inference, each 512×512 image patch generated a **GeoJSON file** containing bounding boxes of detected animals with associated confidence scores.  
Because the inference pipeline directly produced spatial vector outputs, no raster-to-vector or TIFF/PNG conversion was required.

This stage focused on **data consolidation, spatial consistency validation, and quality assurance** to ensure reliable, analysis-ready detection outputs.

---

#### 🧩 Post-Processing Workflow

1. **GeoJSON Consolidation and Merging**  
   - All per-patch detection GeoJSON files were programmatically combined into a single, statewide detection layer.  
   - The process, implemented via a **GeoPandas-based script** (`merge_geojson.py`), performed:
     - Batch reading of all patch-level GeoJSONs from the output directory  
     - Validation of feature geometries and metadata integrity  
     - Enforcement of a unified **coordinate reference system (CRS: EPSG:3857)**  
     - Concatenation into a single GeoDataFrame  
     - Export to both `.geojson` and `.gpkg` formats for interoperability  

2. **Spatial Consistency Verification**  
   - Confirmed alignment between detection geometries and the corresponding basemap tiles.  
   - Verified that all outputs maintained consistent CRS definitions and bounding extents.  

3. **Detection Quality Filtering**  
   - Removed **low-confidence detections** (`confidence < 0.25`).  
   - Eliminated **duplicate and overlapping boxes** created by sliding-window overlaps.  
   - Ensured all surviving features represented unique, valid animal detections.  

4. **Validation Against Ground Reference Data**  
   - Conducted spot-checks comparing detection clusters with verified ground-truth polygons.  
   - Assessed detection density patterns for spatial realism and alignment with expected grazing areas.  

---

#### 📈 Integration with Downstream Analysis

The validated and merged detection layer served as a key spatial input for the  
**“Grazing Activity Classification”** project, where animal distribution data was used to classify regions as  
**Active Grazing Zones** or **Inactive/Low-Activity Areas** based on livestock density metrics.

---

#### 🗂️ Output Artifacts

| File | Description |
|------|--------------|
| `*_predictions.geojson` | Raw YOLOv11n per-patch detections with confidence attributes |
| `merged_predictions.geojson` | Unified statewide detection layer after spatial validation |
| `merged_predictions.gpkg` | GeoPackage version preserving CRS and metadata |
| `grazing_activity_inputs.geojson` | Quality-assured dataset used for Grazing Activity Classification |

---

#### ⚙️ Summary of Post-Processing Objectives

| Objective | Description |
|------------|--------------|
| **CRS Standardization** | All detections validated to EPSG:3857 projection |
| **Data Integration** | Combined 1000+ patch-level GeoJSONs into a single statewide dataset |
| **Error Filtering** | Removed false or redundant detections |
| **Quality Assurance** | Verified accuracy, geometry validity, and alignment |
| **Analytical Readiness** | Prepared data for grazing intensity and density mapping |

---

**In summary:**  
This post-processing and spatial validation stage ensured that YOLOv11n outputs were geometrically accurate, quality-checked, and fully standardized for downstream geospatial modeling and regulatory use within the **LIVESTOCK-EST-CA** framework.

## Configuration Details

### Key Parameters

**Data Generation:**
- `CHIP_SIZE`: 512px square chips (standard for YOLOv8 at high res)
- `NEG_BUFFER_M`: 1m buffer around positive samples for negative mining
- `MAX_WORKERS`: Parallel workers for tile processing (recommend 24-32 on high-core machines)

**Training:**
- `EPOCHS`: 100 (early stopping based on validation mAP)
- `BATCH`: 64 (adjust based on GPU memory)
- `LR0`: 5e-4 (initial learning rate with warmup)
- `WEIGHT_DECAY`: 5e-4 (L2 regularization)
- `IMG_SIZE`: 512 (maintain consistency with chip size)

**Model:**
- YOLOv8m: Best balance of speed and accuracy
- 3-channel input: RGB for enhanced detection
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

# Enable gradient accumulation in YOLOv11
# (set in ultralytics training arguments if needed)
```

### S3 Access Issues

```bash
# Verify credentials
aws s3 ls s3://{BUCKET}/

# Check bucket permissions
aws s3api get-bucket-policy --bucket {BUCKET}

# Test file access
aws s3 cp s3://{BUCKET}/{RGB_PREFIX}/sample.tif ./test.tif

# Verify GDAL VSI access
gdalinfo /vsis3/{BUCKET}/{RGB_PREFIX}/sample.tif
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
