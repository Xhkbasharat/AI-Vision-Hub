
import os
import secrets

from pathlib import Path


# ==========================================
# PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

UPLOAD_FOLDER = DATA_DIR / "uploads"

KNOWN_FACES_FOLDER = DATA_DIR / "known_faces"

MODEL_FOLDER = BASE_DIR / "models"

DATABASE_PATH = DATA_DIR / "ai_vision_hub.db"


# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================

APP_ENV = os.getenv(
    "APP_ENV",
    "development"
).lower()

IS_PRODUCTION = APP_ENV == "production"


# ==========================================
# SECRET KEY
# ==========================================

SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:

    if IS_PRODUCTION:

        raise RuntimeError(
            "SECRET_KEY must be configured in production."
        )

    SECRET_KEY = secrets.token_hex(32)


if len(SECRET_KEY) < 32:

    raise RuntimeError(
        "SECRET_KEY must contain at least 32 characters."
    )


# ==========================================
# FLASK SETTINGS
# ==========================================

DEBUG = (
    os.getenv("FLASK_DEBUG", "false").lower() == "true"
)

if IS_PRODUCTION and DEBUG:

    raise RuntimeError(
        "Debug mode must be disabled in production."
    )


MAX_CONTENT_LENGTH = int(
    os.getenv(
        "MAX_CONTENT_LENGTH",
        str(10 * 1024 * 1024)
    )
)


# ==========================================
# COOKIE SECURITY
# ==========================================

COOKIE_SECURE = (
    os.getenv(
        "COOKIE_SECURE",
        "true" if IS_PRODUCTION else "false"
    ).lower() == "true"
)


# ==========================================
# FILE VALIDATION
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
# FACE RECOGNITION
# ==========================================

FACE_MATCH_THRESHOLD = float(
    os.getenv(
        "FACE_MATCH_THRESHOLD",
        "0.50"
    )
)


# ==========================================
# OBJECT DETECTION
# ==========================================

OBJECT_CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "OBJECT_CONFIDENCE_THRESHOLD",
        "0.50"
    )
)


# ==========================================
# AI MODEL CONFIGURATION
# ==========================================

YOLO_MODEL_NAME = os.getenv(
    "YOLO_MODEL_NAME",
    "yolov8n.pt"
)

IMAGE_CLASSIFICATION_MODEL = os.getenv(
    "IMAGE_CLASSIFICATION_MODEL",
    "MobileNetV2"
)

SENTIMENT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

CHATBOT_MODEL_NAME = os.getenv(
    "CHATBOT_MODEL_NAME",
    "microsoft/DialoGPT-small"
)


# ==========================================
# APPLICATION INFORMATION
# ==========================================

APP_NAME = "AI Vision Hub"

APP_VERSION = "1.0.0"


# ==========================================
# CREATE REQUIRED DIRECTORIES
# ==========================================

REQUIRED_FOLDERS = [
    DATA_DIR,
    UPLOAD_FOLDER,
    KNOWN_FACES_FOLDER,
    MODEL_FOLDER
]

for folder in REQUIRED_FOLDERS:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )
