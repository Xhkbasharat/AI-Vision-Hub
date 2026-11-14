
import os
import secrets

from pathlib import Path

from dotenv import load_dotenv


# ==========================================
# PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

UPLOAD_FOLDER = DATA_DIR / "uploads"

KNOWN_FACES_FOLDER = DATA_DIR / "known_faces"

MODEL_FOLDER = BASE_DIR / "models"

LOG_FOLDER = BASE_DIR / "logs"


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv(
    BASE_DIR / ".env"
)


# ==========================================
# APPLICATION SETTINGS
# ==========================================

APP_NAME = "AI Vision Hub"

APP_VERSION = "1.0.0"

APP_ENV = os.getenv(
    "APP_ENV",
    "development"
).strip().lower()

IS_PRODUCTION = (
    APP_ENV == "production"
)


# ==========================================
# SECRET KEY
# ==========================================

SECRET_KEY = os.getenv(
    "SECRET_KEY"
)

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
# DEBUG SETTINGS
# ==========================================

DEBUG = os.getenv(
    "DEBUG",
    "false" if IS_PRODUCTION else "true"
).strip().lower() in (
    "true",
    "1",
    "yes"
)

if IS_PRODUCTION and DEBUG:

    raise RuntimeError(
        "Debug mode must be disabled in production."
    )


# ==========================================
# UPLOAD SETTINGS
# ==========================================

MAX_CONTENT_LENGTH = int(
    os.getenv(
        "MAX_CONTENT_LENGTH",
        str(10 * 1024 * 1024)
    )
)

if MAX_CONTENT_LENGTH <= 0:

    raise RuntimeError(
        "MAX_CONTENT_LENGTH must be positive."
    )

ALLOWED_IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# ==========================================
# SECURITY SETTINGS
# ==========================================

COOKIE_SECURE = os.getenv(
    "COOKIE_SECURE",
    "true" if IS_PRODUCTION else "false"
).strip().lower() in (
    "true",
    "1",
    "yes"
)

if IS_PRODUCTION and not COOKIE_SECURE:

    raise RuntimeError(
        "Secure cookies must be enabled in production."
    )


# ==========================================
# FACE RECOGNITION SETTINGS
# ==========================================

FACE_MATCH_THRESHOLD = float(
    os.getenv(
        "FACE_MATCH_THRESHOLD",
        "70"
    )
)

if FACE_MATCH_THRESHOLD <= 0:

    raise RuntimeError(
        "FACE_MATCH_THRESHOLD must be positive."
    )


# ==========================================
# OBJECT DETECTION SETTINGS
# ==========================================

OBJECT_CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "OBJECT_CONFIDENCE_THRESHOLD",
        "0.25"
    )
)

if not 0 <= OBJECT_CONFIDENCE_THRESHOLD <= 1:

    raise RuntimeError(
        "OBJECT_CONFIDENCE_THRESHOLD must be between 0 and 1."
    )


# ==========================================
# AI MODEL CONFIGURATION
# ==========================================

SENTIMENT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english"
)

CHATBOT_MODEL_NAME = os.getenv(
    "CHATBOT_MODEL_NAME",
    "microsoft/DialoGPT-small"
)


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

DATABASE_PATH = (
    DATA_DIR / "ai_vision_hub.db"
)


# ==========================================
# CREATE REQUIRED DIRECTORIES
# ==========================================

REQUIRED_DIRECTORIES = [
    DATA_DIR,
    UPLOAD_FOLDER,
    KNOWN_FACES_FOLDER,
    MODEL_FOLDER,
    LOG_FOLDER
]

for directory in REQUIRED_DIRECTORIES:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================
# CONFIGURATION SUMMARY
# ==========================================

def get_config_summary():

    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "environment": APP_ENV,
        "debug": DEBUG,
        "production": IS_PRODUCTION,
        "max_content_length": MAX_CONTENT_LENGTH,
        "allowed_image_extensions": sorted(
            ALLOWED_IMAGE_EXTENSIONS
        ),
        "cookie_secure": COOKIE_SECURE,
        "face_match_threshold": FACE_MATCH_THRESHOLD,
        "object_confidence_threshold": (
            OBJECT_CONFIDENCE_THRESHOLD
        ),
        "database_path": str(DATABASE_PATH)
    }


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "UPLOAD_FOLDER",
    "KNOWN_FACES_FOLDER",
    "MODEL_FOLDER",
    "LOG_FOLDER",
    "DATABASE_PATH",
    "APP_NAME",
    "APP_VERSION",
    "APP_ENV",
    "IS_PRODUCTION",
    "SECRET_KEY",
    "DEBUG",
    "MAX_CONTENT_LENGTH",
    "ALLOWED_IMAGE_EXTENSIONS",
    "COOKIE_SECURE",
    "FACE_MATCH_THRESHOLD",
    "OBJECT_CONFIDENCE_THRESHOLD",
    "SENTIMENT_MODEL_NAME",
    "CHATBOT_MODEL_NAME",
    "get_config_summary"
]
