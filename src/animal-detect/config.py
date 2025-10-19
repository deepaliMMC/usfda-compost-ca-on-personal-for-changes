from pydantic import BaseModel, field_validator
import os
from pathlib import Path


class Settings(BaseModel):
    # -------------------- PATHS --------------------
    DATA_ROOT: Path = Path("/home/sagemaker-user/COMPOST-DETECT-CA/datasets/compost/compost_yoloooo_4band_v1")
    PROJECT_DIR: Path = Path("/home/sagemaker-user/compost yolo/compost_yolo4_single")
    RUN_NAME: str = "yolov8m_4ch"
    PRETRAINED: Path = Path("/home/sagemaker-user/COMPOST-DETECT-CA/src/compost-detect/data/yolov8m.pt")  # local weights
    YOLO_MODEL_PATH:Path = Path("/home/sagemaker-user/COMPOST-DETECT-CA/src/compost-detect/data/model.pt")
    IMAGE_PATH:Path = Path("/home/sagemaker-user/COMPOST-DETECT-CA/test_data/compost/compost_yoloooo_4band_v1")
    # -------------------- TRAINING HYPERPARAMS --------------------
    IMG_SIZE: int = 1024
    EPOCHS: int = 100
    BATCH: int = 12           # fits A10G @ 1024 with 4ch on YOLOv8m
    SAVE_PERIOD: int = 10
    NUM_WORKERS: int = 8      # good balance on g5.24xlarge
    LR0: float = 5e-4
    WEIGHT_DECAY: float = 5e-4
    BUCKET: str = "objectdetction-ca"
    #-------------------- DATA GENERATION --------------------
    NAIP_PREFIX: str = "NAIP_Imagery_2024/"
    POS_GEOJSON: str  = "compost_polygons.geojson"
    NEG_GEOJSON: str  = "neg_points.geojson"
    OUT_PREFIX : str  = "compost_yolo_4band_v1"
    CHIP_SIZE: int    = 1024
    NEG_BUFFER_M: int = 200.0
    POS_CLASS_ID: int = 0
    MAX_WORKERS: int  = int(os.environ.get("MAX_WORKERS", "24"))
    # -------------------- CLASS --------------------
    CLASS_NAMES: list[str] = ["compost"]

    # -------------------- Validators / Init --------------------
    @field_validator("PROJECT_DIR", "DATA_ROOT", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v)

    def create_dirs(self):
        """Ensure project directories exist."""
        self.PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        (self.PROJECT_DIR / self.RUN_NAME).mkdir(parents=True, exist_ok=True)

    model_config = {
        "populate_by_name": True,
    }


def load_settings() -> Settings:
    st = Settings()
    st.create_dirs()
    return st
