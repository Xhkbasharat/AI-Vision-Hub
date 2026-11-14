
import logging
import os
import uuid

from pathlib import Path
from threading import Lock

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    session
)

from PIL import (
    Image,
    UnidentifiedImageError
)

from werkzeug.utils import secure_filename

from config import (
    SECRET_KEY,
    DEBUG,
    MAX_CONTENT_LENGTH,
    ALLOWED_IMAGE_EXTENSIONS,
    UPLOAD_FOLDER,
    COOKIE_SECURE,
    APP_NAME,
    APP_VERSION
)

from database import Database

from modules.face_recognition import FaceRecognition
from modules.object_detection import ObjectDetection
from modules.image_classification import ImageClassification
from modules.ml_prediction import MLPrediction
from modules.sentiment_analysis import SentimentAnalyzer
from modules.fake_news import FakeNewsDetector
from modules.chatbot import Chatbot

from modules.logger import (
    app_logger,
    log_info,
    log_warning,
    log_error
)

from modules.error_handler import (
    AIVisionHubError,
    handle_error
)


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)

app.config.update(
    SECRET_KEY=SECRET_KEY,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=COOKIE_SECURE,
    JSON_SORT_KEYS=False
)


# ==========================================
# LOGGER
# ==========================================

logger = app_logger


# ==========================================
# DATABASE
# ==========================================

db = Database()


# ==========================================
# AI MODEL MANAGEMENT
# ==========================================

model_instances = {}

model_lock = Lock()


MODEL_CLASSES = {
    "face": FaceRecognition,
    "objects": ObjectDetection,
    "classification": ImageClassification,
    "ml": MLPrediction,
    "sentiment": SentimentAnalyzer,
    "chatbot": Chatbot
}


def get_model(name):
    """
    Load an AI model only when required.
    Reuse the instance for future requests.
    """

    if name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {name}")

    with model_lock:

        if name not in model_instances:

            logger.info(
                "Loading AI model: %s",
                name
            )

            try:

                model_instances[name] = MODEL_CLASSES[name]()

                logger.info(
                    "AI model loaded: %s",
                    name
                )

            except Exception:

                logger.exception(
                    "Could not load AI model: %s",
                    name
                )

                raise

        return model_instances[name]


# ==========================================
# API RESPONSE HELPERS
# ==========================================

def success_response(
    result,
    message="Operation completed successfully."
):

    return jsonify({
        "success": True,
        "message": message,
        "result": result
    })


def error_response(
    message,
    status_code=400,
    code="APPLICATION_ERROR"
):

    return jsonify({
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }), status_code


# ==========================================
# IMAGE VALIDATION
# ==========================================

def save_image(file):
    """
    Validate and save an uploaded image.
    """

    if file is None or not file.filename:

        raise ValueError(
            "Please select an image."
        )

    filename = secure_filename(file.filename)

    if not filename or "." not in filename:

        raise ValueError(
            "Invalid filename."
        )

    extension = filename.rsplit(".", 1)[1].lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:

        raise ValueError(
            "Unsupported image format."
        )

    try:

        file.stream.seek(0)

        with Image.open(file.stream) as image:

            image.verify()

        file.stream.seek(0)

        with Image.open(file.stream) as image:

            width, height = image.size

            if width <= 0 or height <= 0:

                raise ValueError(
                    "Invalid image dimensions."
                )

            if width * height > 25_000_000:

                raise ValueError(
                    "Image dimensions are too large."
                )

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError
    ) as error:

        raise ValueError(
            "The uploaded file is not a valid or safe image."
        ) from error

    unique_filename = (
        f"{uuid.uuid4().hex}.{extension}"
    )

    upload_folder = Path(UPLOAD_FOLDER)

    upload_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = upload_folder / unique_filename

    file.stream.seek(0)

    file.save(str(image_path))

    try:

        db.add_upload(
            unique_filename,
            "image"
        )

        logger.info(
            "Image uploaded successfully: %s",
            unique_filename
        )

    except Exception:

        image_path.unlink(missing_ok=True)

        logger.exception(
            "Could not record uploaded image."
        )

        raise

    return str(image_path)


# ==========================================
# HOME DASHBOARD
# ==========================================

@app.route("/", methods=["GET"])
def home():

    try:

        stats = db.get_dashboard_stats()

        return render_template(
            "index.html",
            stats=stats
        )

    except Exception:

        logger.exception(
            "Dashboard loading failed."
        )

        return "Dashboard unavailable.", 500


# ==========================================
# OBJECT DETECTION
# ==========================================

@app.route("/detect-objects", methods=["POST"])
def detect_objects():

    try:

        image_path = save_image(
            request.files.get("image")
        )

        detector = get_model("objects")

        results = detector.detect_objects(
            image_path
        )

        db.save_analysis(
            module_name="object_detection",
            input_text=Path(image_path).name,
            result=str(results)
        )

        return success_response(
            results,
            "Object detection completed."
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        logger.exception(
            "Object detection failed."
        )

        return error_response(
            "Object detection failed.",
            500
        )


# ==========================================
# IMAGE CLASSIFICATION
# ==========================================

@app.route("/classify-image", methods=["POST"])
def classify_image():

    try:

        image_path = save_image(
            request.files.get("image")
        )

        classifier = get_model("classification")

        results = classifier.classify_image(
            image_path
        )

        db.save_analysis(
            module_name="image_classification",
            input_text=Path(image_path).name,
            result=str(results)
        )

        return success_response(
            results,
            "Image classification completed."
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        logger.exception(
            "Image classification failed."
        )

        return error_response(
            "Image classification failed.",
            500
        )


# ==========================================
# FACE REGISTRATION
# ==========================================

@app.route("/register-face", methods=["POST"])
def register_face():

    try:

        person_name = request.form.get(
            "person_name",
            ""
        ).strip()

        if not person_name:

            return error_response(
                "Please enter a person's name."
            )

        if len(person_name) > 50:

            return error_response(
                "Name must not exceed 50 characters."
            )

        image_path = save_image(
            request.files.get("image")
        )

        face_system = get_model("face")

        result = face_system.register_face(
            person_name,
            image_path
        )

        return success_response(
            result,
            "Face registration completed."
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        logger.exception(
            "Face registration failed."
        )

        return error_response(
            "Face registration failed.",
            500
        )


# ==========================================
# TRAIN FACE RECOGNITION
# ==========================================

@app.route("/train-faces", methods=["POST"])
def train_faces():

    try:

        face_system = get_model("face")

        result = face_system.train_model()

        return success_response(
            result,
            "Face recognition training completed."
        )

    except Exception:

        logger.exception(
            "Face training failed."
        )

        return error_response(
            "Face training failed.",
            500
        )


# ==========================================
# FACE RECOGNITION
# ==========================================

@app.route("/recognize-face", methods=["POST"])
def recognize_face():

    try:

        image_path = save_image(
            request.files.get("image")
        )

        face_system = get_model("face")

        result = face_system.recognize_face(
            image_path
        )

        db.save_analysis(
            module_name="face_recognition",
            input_text=Path(image_path).name,
            result=str(result)
        )

        return success_response(
            result,
            "Face recognition completed."
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        logger.exception(
            "Face recognition failed."
        )

        return error_response(
            "Face recognition failed.",
            500
        )


# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

@app.route("/sentiment", methods=["POST"])
def sentiment():

    try:

        data = request.get_json(silent=True) or {}

        text = data.get(
            "text",
            ""
        )

        if not isinstance(text, str):

            return error_response(
                "Text must be a string."
            )

        text = text.strip()

        if not text:

            return error_response(
                "Please enter text to analyze."
            )

        if len(text) > 5000:

            return error_response(
                "Text must not exceed 5000 characters."
            )

        analyzer = get_model("sentiment")

        result = analyzer.analyze_sentiment(text)

        db.save_analysis(
            module_name="sentiment_analysis",
            input_text=text,
            result=str(result),
            confidence=result.get("confidence")
        )

        return success_response(
            result,
            "Sentiment analysis completed."
        )

    except Exception:

        logger.exception(
            "Sentiment analysis failed."
        )

        return error_response(
            "Sentiment analysis failed.",
            500
        )


# ==========================================
# FAKE NEWS DETECTION
# ==========================================

@app.route("/fake-news", methods=["POST"])
def fake_news():

    try:

        data = request.get_json(silent=True) or {}

        text = data.get(
            "text",
            ""
        )

        if not isinstance(text, str):

            return error_response(
                "Article text must be a string."
            )

        text = text.strip()

        if not text:

            return error_response(
                "Please enter an article."
            )

        if len(text) > 10000:

            return error_response(
                "Article must not exceed 10000 characters."
            )

        detector = FakeNewsDetector()

        result = detector.predict(text)

        db.save_analysis(
            module_name="fake_news",
            input_text=text[:1000],
            result=str(result),
            confidence=result.get("confidence")
        )

        return success_response(
            result,
            "Fake-news analysis completed."
        )

    except FileNotFoundError:

        return error_response(
            "Fake-news model not found. Train the model first.",
            503
        )

    except Exception:

        logger.exception(
            "Fake-news detection failed."
        )

        return error_response(
            "Fake-news detection failed.",
            500
        )


# ==========================================
# MACHINE LEARNING PREDICTION
# ==========================================

@app.route("/ml-prediction", methods=["POST"])
def ml_prediction():

    try:

        data = request.get_json(silent=True) or {}

        values = data.get("values")

        model_name = data.get(
            "model",
            "random_forest"
        )

        if not isinstance(values, list) or len(values) != 4:

            return error_response(
                "Exactly four numeric feature values are required."
            )

        if model_name not in [
            "random_forest",
            "xgboost"
        ]:

            return error_response(
                "Unsupported prediction model."
            )

        predictor = get_model("ml")

        result = predictor.predict(
            values,
            model_name=model_name
        )

        db.save_analysis(
            module_name="ml_prediction",
            input_text=str(values),
            result=str(result),
            confidence=result.get("confidence")
        )

        return success_response(
            result,
            "Prediction completed."
        )

    except (ValueError, TypeError) as error:

        return error_response(
            str(error)
        )

    except Exception:

        logger.exception(
            "Machine learning prediction failed."
        )

        return error_response(
            "Machine learning prediction failed.",
            500
        )


# ==========================================
# CHATBOT
# ==========================================

@app.route("/chatbot", methods=["POST"])
def chatbot():

    try:

        message = request.form.get(
            "message",
            ""
        ).strip()

        if not message:

            return error_response(
                "Please enter a message."
            )

        if len(message) > 1000:

            return error_response(
                "Message must not exceed 1000 characters."
            )

        bot = get_model("chatbot")

        history = session.get(
            "chat_history",
            []
        )

        result = bot.chat(
            message,
            history=history
        )

        session["chat_history"] = result.get(
            "history",
            []
        )[-4:]

        session.modified = True

        return success_response(
            result,
            "Chatbot response generated."
        )

    except Exception:

        logger.exception(
            "Chatbot request failed."
        )

        return error_response(
            "Chatbot request failed.",
            500
        )


# ==========================================
# RESET CHAT
# ==========================================

@app.route("/reset-chat", methods=["POST"])
def reset_chat():

    session.pop(
        "chat_history",
        None
    )

    return success_response(
        {"reset": True},
        "Chat history cleared."
    )


# ==========================================
# ANALYSIS HISTORY
# ==========================================

@app.route("/analysis-history", methods=["GET"])
def analysis_history():

    try:

        limit = request.args.get(
            "limit",
            default=50,
            type=int
        )

        history = db.get_analysis_history(
            limit
        )

        return success_response(
            history,
            "Analysis history retrieved."
        )

    except Exception:

        logger.exception(
            "Analysis history retrieval failed."
        )

        return error_response(
            "Could not retrieve analysis history.",
            500
        )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "application": APP_NAME,
        "version": APP_VERSION
    }), 200


# ==========================================
# REQUEST TOO LARGE
# ==========================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return error_response(
        "Uploaded file is too large.",
        413,
        "FILE_TOO_LARGE"
    )


# ==========================================
# GLOBAL ERROR HANDLER
# ==========================================

@app.errorhandler(500)
def internal_server_error(error):

    logger.exception(
        "Internal server error."
    )

    return error_response(
        "An internal server error occurred.",
        500,
        "INTERNAL_SERVER_ERROR"
    )


# ==========================================
# APPLICATION STARTUP
# ==========================================

if __name__ == "__main__":

    logger.info(
        "Starting %s version %s",
        APP_NAME,
        APP_VERSION
    )

    logger.info(
        "Debug mode: %s",
        DEBUG
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=DEBUG
    )
