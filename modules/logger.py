
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ==========================================
# LOG DIRECTORY
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FILE = LOG_DIR / "ai_vision_hub.log"


# ==========================================
# LOGGER CONFIGURATION
# ==========================================

def setup_logger(
    name="AI-Vision-Hub",
    level=logging.INFO
):

    logger = logging.getLogger(name)

    logger.setLevel(level)

    # Avoid adding duplicate handlers.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    # ======================================
    # FILE HANDLER
    # ======================================

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # ======================================
    # CONSOLE HANDLER
    # ======================================

    console_handler = logging.StreamHandler()

    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # ======================================
    # ATTACH HANDLERS
    # ======================================

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


# ==========================================
# DEFAULT APPLICATION LOGGER
# ==========================================

app_logger = setup_logger()


# ==========================================
# CONVENIENCE FUNCTIONS
# ==========================================

def log_info(message):

    app_logger.info(message)


def log_warning(message):

    app_logger.warning(message)


def log_error(message, exc_info=False):

    app_logger.error(
        message,
        exc_info=exc_info
    )


def log_debug(message):

    app_logger.debug(message)


# ==========================================
# PUBLIC EXPORTS
# ==========================================

__all__ = [
    "setup_logger",
    "app_logger",
    "log_info",
    "log_warning",
    "log_error",
    "log_debug",
    "LOG_FILE"
]
