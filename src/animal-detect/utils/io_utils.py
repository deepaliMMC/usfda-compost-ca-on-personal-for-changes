import os
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from utils.config import load_settings
settings = load_settings()

def write_geotiff(path, src_profile, arr, transform):
    profile = src_profile.copy()
    profile.update({
        "driver": "GTiff",
        "height": arr.shape[1],
        "width":  arr.shape[2],
        "count":  arr.shape[0],
        "dtype":  arr.dtype,
        "transform": transform,
        "compress": "lzw"
    })
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)



def raster_bbox_to_yolo(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    H, W = mask.shape
    bw = (xmax - xmin + 1)
    bh = (ymax - ymin + 1)
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return (cx / W, cy / H, bw / W, bh / H)



def save_label_txt(path, yolo_boxes, class_id=0):
    with open(path, "w") as f:
        for (cx, cy, w, h) in yolo_boxes:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            

def write_dataset_yaml():
    txt = (
        f"path: s3://{settings.BUCKET}/{settings.OUT_PREFIX}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: [compost]\n"
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tf:
        tf.write(txt); tmp = tf.name
    s3.upload_file(tmp, settings.BUCKET, f"{settings.OUT_PREFIX}/dataset.yaml")
    os.remove(tmp)