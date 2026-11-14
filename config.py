
import os
from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_FOLDER = DATA_DIR / "uploads"
KNOWN_FACES_FOLDER = DATA_DIR / "known_faces"

MODEL_FOLDER = BASE_DIR / "models"

DATABASE_PATH = DATA_DIR / "ai_vision_hub.db"


# ==========================================
# AUTOMATIC DIRECTORY CREATION
# ==========================================

REQUIRED_FOLDERS = [
    DATA_DIR,
    UPLOAD_FOLDER,
    KNOWN_FACES_FOLDER,
    MODEL_FOLDER
]

for folder in REQUIRED_FOLDERS:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# FLASK CONFIGURATION
# ==========================================

SECRET_KEY = os.environ.get("SECRET_KEY")

if not SECRET_KEY:
    SECRET_KEY = "development-only-change-this-secret"

DEBUG = (
    os.environ.get("FLASK_DEBUG", "false").lower() == "true"
)

MAX_CONTENT_LENGTH = 10 * 1024 * 1024


# ==========================================
# UPLOAD CONFIGURATION
# ==========================================

ALLOWED_IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "avi",
    "mov"
}


# ==========================================
# AI MODEL CONFIGURATION
# ==========================================

YOLO_MODEL_NAME = "yolov8n.pt"

IMAGE_CLASSIFICATION_MODEL = "MobileNetV2"

SENTIMENT_MODEL_NAME = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

CHATBOT_MODEL_NAME = "microsoft/DialoGPT-small"


# ==========================================
# MODEL THRESHOLDS
# ==========================================

FACE_MATCH_THRESHOLD = 0.50

OBJECT_CONFIDENCE_THRESHOLD = 0.50


# ==========================================
# APPLICATION INFORMATION
# ==========================================

APP_NAME = "AI Vision Hub"

APP_VERSION = "1.0.0"
