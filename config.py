
import os
from pathlib import Path

# Main project directory
BASE_DIR = Path(__file__).resolve().parent

# Project directories
DATA_DIR = BASE_DIR / "data"
UPLOAD_FOLDER = DATA_DIR / "uploads"
KNOWN_FACES_FOLDER = DATA_DIR / "known_faces"
MODEL_FOLDER = BASE_DIR / "models"

# Database
DATABASE_PATH = DATA_DIR / "ai_vision_hub.db"

# Create required directories automatically
for folder in [
    DATA_DIR,
    UPLOAD_FOLDER,
    KNOWN_FACES_FOLDER,
    MODEL_FOLDER
]:
    folder.mkdir(parents=True, exist_ok=True)

# Flask settings
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "development-key-change-before-deployment"
)

DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

# Upload settings
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

ALLOWED_IMAGE_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp"
}

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4", "avi", "mov"
}

# AI model settings
YOLO_MODEL_NAME = "yolov8n.pt"

IMAGE_CLASSIFICATION_MODEL = (
    "MobileNetV2"
)

SENTIMENT_MODEL_NAME = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

CHATBOT_MODEL_NAME = (
    "microsoft/DialoGPT-small"
)

# Face recognition settings
FACE_MATCH_THRESHOLD = 0.50

# Object detection settings
OBJECT_CONFIDENCE_THRESHOLD = 0.50

# Application settings
APP_NAME = "AI Vision Hub"
APP_VERSION = "1.0.0"
