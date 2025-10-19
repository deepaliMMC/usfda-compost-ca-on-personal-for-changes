import boto3
from typing import List
from .config import BUCKET

s3 = boto3.client("s3")

def list_naip_tifs(bucket: str = BUCKET, prefix: str = "") -> List[str]:
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

def upload_file(local_path: str, s3_key: str):
    s3.upload_file(local_path, BUCKET, s3_key)
