import boto3
from pathlib import Path
from typing import List
from utils.config import load_settings
settings = load_settings()

s3 = boto3.client("s3")

def list_naip_tifs(bucket: str, prefix: str) -> List[str]:
    keys = []
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if not k.lower().endswith(".tif"):
                continue
            if "/" in k[len(prefix):]:
                continue
            keys.append(k)
    return sorted(keys)




def tile_already_processed(tile_stem: str) -> bool:
    prefixes = [
        f"{settings.OUT_PREFIX}/images/train/pos_{tile_stem}_",
        f"{settings.OUT_PREFIX}/images/val/pos_{tile_stem}_",
        f"{settings.OUT_PREFIX}/images/train/neg_{tile_stem}_",
        f"{settings.OUT_PREFIX}/images/val/neg_{tile_stem}_",
    ]
    for pref in prefixes:
        resp = s3.list_objects_v2(Bucket=settings.BUCKET, Prefix=pref, MaxKeys=1)
        if resp.get("KeyCount", 0) > 0:
            return True
    return False