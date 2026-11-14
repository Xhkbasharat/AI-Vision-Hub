
import logging
from datetime import datetime, timezone


# ==========================================
# LOGGER CONFIGURATION
# ==========================================

logger = logging.getLogger("AI-Vision-Hub")


# ==========================================
# CUSTOM APPLICATION EXCEPTIONS
# ==========================================

class AIVisionHubError(Exception):
    """Base exception for AI Vision Hub."""

    def __init__(self, message, code="APPLICATION_ERROR"):
        super().__init__(message)

        self.message = message
        self.code = code


class ImageValidationError(AIVisionHubError):
    """Raised when an uploaded image is invalid."""

    def __init__(self, message="Invalid image file."):
        super().__init__(
            message,
            code="IMAGE_VALIDATION_ERROR"
        )


class ModelLoadingError(AIVisionHubError):
    """Raised when an AI model cannot be loaded."""

    def __init__(self, message="AI model could not be loaded."):
        super().__init__(
            message,
            code="MODEL_LOADING_ERROR"
        )


class PredictionError(AIVisionHubError):
    """Raised when an AI prediction fails."""

    def __init__(self, message="Prediction failed."):
        super().__init__(
            message,
            code="PREDICTION_ERROR"
        )


class DatabaseError(AIVisionHubError):
    """Raised when a database operation fails."""

    def __init__(self, message="Database operation failed."):
        super().__init__(
            message,
            code="DATABASE_ERROR"
        )


# ==========================================
# ERROR LOGGING
# ==========================================

def log_error(error, module_name="application"):

    logger.exception(
        "Error in module: %s | Error: %s",
        module_name,
        error
    )


# ==========================================
# ERROR RESPONSE DATA
# ==========================================

def create_error_response(
    message,
    code="APPLICATION_ERROR"
):

    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat()
        }
    }


# ==========================================
# SAFE ERROR HANDLING
# ==========================================

def handle_error(error, module_name="application"):

    """
    Log the original error and return a safe
    message that can be shown to the user.
    """

    log_error(
        error,
        module_name
    )

    if isinstance(error, AIVisionHubError):

        return create_error_response(
            error.message,
            error.code
        )

    return create_error_response(
        "An unexpected error occurred. Please try again.",
        "INTERNAL_ERROR"
    )


# ==========================================
# ERROR MESSAGE VALIDATION
# ==========================================

def validate_error_message(message):

    if not isinstance(message, str):
        raise TypeError("Error message must be a string.")

    message = message.strip()

    if not message:
        raise ValueError("Error message cannot be empty.")

    return message


# ==========================================
# PUBLIC EXPORTS
# ==========================================

__all__ = [
    "AIVisionHubError",
    "ImageValidationError",
    "ModelLoadingError",
    "PredictionError",
    "DatabaseError",
    "log_error",
    "create_error_response",
    "handle_error",
    "validate_error_message"
]
