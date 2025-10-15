Action: Create new file with this content:
# Complete Reproducibility Guide - COMPOST-DETECT-CA

This guide ensures exact reproduction of compost facility detection model training and inference for FDA compliance.

## 1. Environment Setup

### Option 1: Conda (Recommended)

\`\`\`bash
# Create environment
conda create -n compost-detect python=3.10
conda activate compost-detect

# Install dependencies
pip install -r requirements.txt

# Verify installation (MUST PASS for FDA)
python scripts/verify_installation.py
# Expected: ✓ FDA VERIFICATION PASSED
\`\`\`

### Option 2: Docker (Exact Reproducibility)

\`\`\`bash
# Build image
docker build -t compost-detect:v2.0 .

# Run container
docker run --gpus all -it \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  compost-detect:v2.0
\`\`\`

## 2. Data Preparation

### Step 1: Obtain NAIP Imagery

Contact USDA Farm Service Agency or use AWS Open Data:
- **Source:** https://registry.opendata.aws/naip/
- **Resolution:** 0.6m (4-band: RGB + NIR)
- **Coverage:** California statewide
- **Format:** Cloud-Optimized GeoTIFF (COG)

### Step 2: Prepare Annotation Files

**Positive Samples (compost_polygons.geojson):**
- Polygon geometries of known compost facilities
- Minimum 500 facilities recommended
- CRS: EPSG:4326 (WGS84)

**Negative Samples (neg_points.geojson):**
- Point locations of non-compost areas
- Minimum 1000 points recommended
- Must be >200m away from positive samples

### Step 3: Verify Data Structure

\`\`\`bash
python scripts/verify_data_structure.py \
  --pos-geojson data/compost_polygons.geojson \
  --neg-geojson data/neg_points.geojson \
  --s3-bucket objectdetction-ca

# Expected: ✓ FDA DATA VERIFICATION PASSED
\`\`\`

## 3. Data Generation (Create Training Chips)

\`\`\`bash
# Configure S3 and paths in configs/default.yaml
# Then run:
python src/compost-detect/data_generation.py

# Expected Output:
# - Generated ~12,000 chips (1024×1024×4)
# - Train/val split by tile location
# - YOLO format labels created
# - Uploaded to S3: s3://bucket/compost_yolo_4band_v1/
\`\`\`

**Expected Statistics:**
- Total chips: 10,000-15,000
- Training chips: ~70-80%
- Validation chips: ~20-30%
- Positive chips: ~50%
- Negative chips: ~50%

## 4. Model Training

\`\`\`bash
# Set random seed for reproducibility
export PYTHONHASHSEED=42

# Train YOLOv8m
python src/compost-detect/training.py

# Training time: ~8-12 hours on ml.g5.xlarge (AWS)
# Expected GPU memory: 18-20 GB
\`\`\`

**Expected Training Curves:**
- Epoch 0: Loss ~1.5
- Epoch 50: Loss ~0.3
- Epoch 100: Loss ~0.15

**Expected Final Metrics (±0.05 variation):**
- mAP@0.5: 0.75-0.80
- mAP@0.5:0.95: 0.60-0.65
- Precision: 0.80-0.85
- Recall: 0.75-0.80

## 5. Model Validation

\`\`\`bash
# Benchmark against FDA targets
python scripts/benchmark_model.py \
  --model runs/detect/yolov8m_4ch/weights/best.pt \
  --data data/compost_yolo_4band_v1/dataset.yaml

# Expected: ✓ FDA PERFORMANCE TARGETS MET
\`\`\`

## 6. Inference on New Imagery

\`\`\`bash
# Run predictions
python src/compost-detect/prediction.py

# Expected: Detections with confidence scores >0.25
\`\`\`

## 7. Verification Checklist

Before FDA submission, verify:

- [ ] All dependencies match pinned versions
- [ ] `verify_installation.py` passes
- [ ] `verify_data_structure.py` passes
- [ ] Training metrics match expected ranges
- [ ] `benchmark_model.py` passes all FDA targets
- [ ] Inference produces reasonable detections

## Troubleshooting

**Issue: verify_installation.py fails**
- Solution: Run `pip install -r requirements.txt --force-reinstall`

**Issue: CUDA not available**
- Solution: Install correct PyTorch for your CUDA version
- Check: https://pytorch.org/get-started/locally/

**Issue: S3 access denied**
- Solution: Run `aws configure` and check credentials

**Issue: Training metrics below FDA targets**
- Solution: Check data quality, increase training epochs, or collect more annotations

## Support

- **Technical Questions:** info@mapmycrop.com
- **GitHub Issues:** https://github.com/Map-My-Crop-Platform/COMPOST-DETECT-CA/issues
- **FDA Compliance:** See docs/fda_compliance.md
