
import logging
import secrets

from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session
)

from werkzeug.utils import secure_filename

from PIL import Image, UnidentifiedImageError

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


# ======================================
# APPLICATION CONFIGURATION
# ======================================

app = Flask(__name__)

app.config.update(
    SECRET_KEY=SECRET_KEY,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=COOKIE_SECURE,
    SESSION_COOKIE_SAMESITE="Lax"
)

app.logger.handlers = app_logger.handlers
app.logger.setLevel(app_logger.level)

database = Database()

# Cache expensive models after initialization.
model_cache = {}


# ======================================
# API RESPONSE HELPERS
# ======================================

def success_response(
    result=None,
    message="Operation completed successfully.",
    status_code=200
):

    return jsonify({
        "success": True,
        "message": message,
        "result": result
    }), status_code


def error_response(
    message,
    status_code=400,
    code="REQUEST_ERROR"
):

    return jsonify({
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }), status_code


# ======================================
# MODEL LOADER
# ======================================

def get_model(model_name):

    model_classes = {
        "face": FaceRecognition,
        "objects": ObjectDetection,
        "classification": ImageClassification,
        "ml": MLPrediction,
        "sentiment": SentimentAnalyzer,
        "chatbot": Chatbot
    }

    if model_name not in model_classes:

        raise ValueError(
            "Unknown model requested."
        )

    if model_name not in model_cache:

        log_info(
            f"Loading model: {model_name}"
        )

        model_cache[model_name] = (
            model_classes[model_name]()
        )

    return model_cache[model_name]


# ======================================
# IMAGE UPLOAD VALIDATION
# ======================================

def save_image(file):

    if not file or not file.filename:

        raise ValueError(
            "Please select an image."
        )

    original_filename = secure_filename(
        file.filename
    )

    if not original_filename:

        raise ValueError(
            "Invalid filename."
        )

    extension = Path(
        original_filename
    ).suffix.lower().lstrip(".")

    if extension not in ALLOWED_IMAGE_EXTENSIONS:

        raise ValueError(
            "Unsupported image format."
        )

    upload_directory = Path(
        UPLOAD_FOLDER
    )

    upload_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # Verify image content rather than trusting extension.
    try:

        image = Image.open(file.stream)

        image.verify()

        file.stream.seek(0)

        image = Image.open(file.stream)

        image_width, image_height = image.size

        if image_width <= 0 or image_height <= 0:

            raise ValueError(
                "Invalid image dimensions."
            )

        if image_width * image_height > 25_000_000:

            raise ValueError(
                "Image dimensions exceed the permitted limit."
            )

        image.verify()

        file.stream.seek(0)

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError
    ) as error:

        raise ValueError(
            "Uploaded file is not a valid image."
        ) from error

    filename = (
        f"{secrets.token_hex(16)}.{extension}"
    )

    destination = (
        upload_directory / filename
    )

    file.save(destination)

    try:

        database.add_upload(
            filename,
            extension
        )

    except Exception:

        destination.unlink(
            missing_ok=True
        )

        raise

    log_info(
        f"Image uploaded successfully: {filename}"
    )

    return str(destination)


# ======================================
# SAVE ANALYSIS
# ======================================

def record_analysis(
    module_name,
    input_text,
    result
):

    confidence = None

    if isinstance(result, dict):

        confidence = result.get(
            "confidence"
        )

    database.save_analysis(
        module_name,
        input_text,
        result,
        confidence
    )


# ======================================
# DASHBOARD
# ======================================

@app.route("/", methods=["GET"])
def index():

    try:

        stats = database.get_dashboard_stats()

        return render_template(
            "index.html",
            stats=stats,
            app_name=APP_NAME,
            app_version=APP_VERSION
        )

    except Exception:

        log_error(
            "Dashboard loading failed."
        )

        return render_template(
            "index.html",
            stats={
                "total_uploads": 0,
                "registered_faces": 0,
                "total_analyses": 0
            },
            app_name=APP_NAME,
            app_version=APP_VERSION
        )


# ======================================
# OBJECT DETECTION
# ======================================

@app.route(
    "/detect-objects",
    methods=["POST"]
)
def detect_objects():

    try:

        image_path = save_image(
            request.files.get("image")
        )

        detector = get_model("objects")

        result = detector.detect_objects(
            image_path
        )

        record_analysis(
            "object_detection",
            Path(image_path).name,
            result
        )

        return success_response(
            result,
            "Object detection completed."
        )

    except ValueError as error:

        return error_response(
            str(error),
            400
        )

    except Exception:

        log_error(
            "Object detection request failed."
        )

        return error_response(
            "Unable to process object detection.",
            500,
            "OBJECT_DETECTION_ERROR"
        )


# ======================================
# IMAGE CLASSIFICATION
# ======================================

@app.route(
    "/classify-image",
    methods=["POST"]
)
def classify_image():

    try:

        image_path = save_image(
            request.files.get("image")
        )

        classifier = get_model(
            "classification"
        )

        result = classifier.classify_image(
            image_path
        )

        record_analysis(
            "image_classification",
            Path(image_path).name,
            result
        )

        return success_response(
            result,
            "Image classification completed."
        )

    except ValueError as error:

        return error_response(
            str(error),
            400
        )

    except Exception:

        log_error(
            "Image classification request failed."
        )

        return error_response(
            "Unable to classify the image.",
            500,
            "IMAGE_CLASSIFICATION_ERROR"
        )


# ======================================
# REGISTER FACE
# ======================================

@app.route(
    "/register-face",
    methods=["POST"]
)
def register_face():

    try:

        person_name = request.form.get(
            "person_name",
            ""
        ).strip()

        if not person_name:

            raise ValueError(
                "Person name is required."
            )

        image_path = save_image(
            request.files.get("image")
        )

        face_system = get_model("face")

        result = face_system.register_face(
            person_name,
            image_path
        )

        record_analysis(
            "face_registration",
            person_name,
            result
        )

        return success_response(
            result,
            "Face registered successfully."
        )

    except ValueError as error:

        return error_response(
            str(error),
            400
        )

    except Exception:

        log_error(
            "Face registration request failed."
        )

        return error_response(
            "Unable to register the face.",
            500,
            "FACE_REGISTRATION_ERROR"
        )


# ======================================
# TRAIN FACE MODEL
# ======================================

@app.route(
    "/train-faces",
    methods=["POST"]
)
def train_faces():

    try:

        face_system = get_model("face")

        result = face_system.train_model()

        record_analysis(
            "face_training",
            "Registered face dataset",
            result
        )

        return success_response(
            result,
            "Face model training completed."
        )

    except ValueError as error:

        return error_response(
            str(error),
            400
        )

    except Exception:

        log_error(
            "Face model training failed."
        )

        return error_response(
            "Unable to train the face recognition model.",
            500,
            "FACE_TRAINING_ERROR"
        )


# ======================================
# RECOGNIZE FACE
# ======================================

@app.route(
    "/recognize-face",
    methods=["POST"]
)
def recognize_face():

    try:

        image_path = save_image(
            request.files.get("image")
        )

        face_system = get_model("face")

        result = face_system.recognize_face(
            image_path
        )

        record_analysis(
            "face_recognition",
            Path(image_path).name,
            result
        )

        return success_response(
            result,
            "Face recognition completed."
        )

    except ValueError as error:

        return error_response(
            str(error),
            400
        )

    except Exception:

        log_error(
            "Face recognition request failed."
        )

        return error_response(
            "Unable to recognize faces.",
            500,
            "FACE_RECOGNITION_ERROR"
        )


# ======================================
# SENTIMENT ANALYSIS
# ======================================

@app.route(
    "/sentiment",
    methods=["POST"]
)
def sentiment():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        text = data.get(
            "text",
            ""
        )

        if not isinstance(text, str):

            raise ValueError(
                "Input must be text."
            )

        if len(text) > 5000:

            raise ValueError(
                "Text cannot exceed 5000 characters."
            )

        analyzer = get_model(
            "sentiment"
        )

        result = analyzer.analyze_sentiment(
            text
        )

        record_analysis(
            "sentiment_analysis",
            text,
            result
        )

        return success_response(
            result,
            "Sentiment analysis completed."
        )

    except ValueError as error:

        return error_response(
            str(error),
            400
        )

    except Exception:

        log_error(
            "Sentiment analysis request failed."
        )

        return error_response(
            "Unable to analyze sentiment.",
            500,
            "SENTIMENT_ERROR"
        )


# ======================================
# FAKE NEWS DETECTION
# ======================================

@app.route(
    "/fake-news",
    methods=["POST"]
)
def fake_news():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        text = data.get(
            "text",
            ""
        )

        if not isinstance(text, str):

            raise ValueError(
                "News content must be text."
            )

        if len(text) > 10000:

            raise ValueError(
                "News content cannot exceed 10000 characters."
            )

        detector = FakeNewsDetector()

        result = detector.predict(
            text
        )

        record_analysis(
            "fake_news_detection",
            text,
            result
        )

        return success_response(
            result,
            "News classification completed."
        )

    except ValueError as error:

        return error_response(
            str(error),
            400
        )

    except FileNotFoundError:

        return error_response(
            "Fake news model is not trained yet.",
            503,
            "MODEL_NOT_AVAILABLE"
        )

    except Exception:

        log_error(
            "Fake news detection request failed."
        )

        return error_response(
            "Unable to classify the news content.",
            500,
            "FAKE_NEWS_ERROR"
        )


# ======================================
# MACHINE LEARNING PREDICTION
# ======================================

@app.route(
    "/ml-prediction",
    methods=["POST"]
)
def ml_prediction():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        values = data.get(
            "values"
        )

        model_name = data.get(
            "model",
            "random_forest"
        )

        if model_name not in [
            "random_forest",
            "xgboost"
        ]:

            raise ValueError(
                "Unsupported machine-learning model."
            )

        predictor = get_model("ml")

        result = predictor.predict(
            values,
            model_name=model_name
        )

        record_analysis(
            "ml_prediction",
            str(values),
            result
        )

        return success_response(
            result,
            "Prediction completed."
        )

    except ValueError as error:

        return error_response(
            str(error),
            400
        )

    except Exception:

        log_error(
            "Machine-learning prediction failed."
        )

        return error_response(
            "Unable to generate prediction.",
            500,
            "ML_PREDICTION_ERROR"
        )


# ======================================
# CHATBOT
# ======================================

@app.route(
    "/chatbot",
    methods=["POST"]
)
def chatbot():

    try:

        message = request.form.get(
            "message",
            ""
        ).strip()

        if not message:

            raise ValueError(
                "Please enter a message."
            )

        if len(message) > 1000:

            raise ValueError(
                "Message cannot exceed 1000 characters."
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

        record_analysis(
            "chatbot",
            message,
            {
                "response": result.get("response")
            }
        )

        return success_response(
            result,
            "Chatbot response generated."
        )

    except ValueError as error:

        return error_response(
            str(error),
            400
        )

    except Exception:

        log_error(
            "Chatbot request failed."
        )

        return error_response(
            "Unable to generate chatbot response.",
            500,
            "CHATBOT_ERROR"
        )


# ======================================
# RESET CHAT
# ======================================

@app.route(
    "/reset-chat",
    methods=["POST"]
)
def reset_chat():

    session.pop(
        "chat_history",
        None
    )

    session.modified = True

    return success_response(
        {
            "history": []
        },
        "Conversation reset successfully."
    )


# ======================================
# ANALYSIS HISTORY
# ======================================

@app.route(
    "/analysis-history",
    methods=["GET"]
)
def analysis_history():

    try:

        raw_limit = request.args.get(
            "limit",
            "50"
        )

        try:

            limit = int(raw_limit)

        except ValueError:

            raise ValueError(
                "Limit must be a valid integer."
            )

        if not 1 <= limit <= 500:

            raise ValueError(
                "Limit must be between 1 and 500."
            )

        history = database.get_analysis_history(
            limit=limit
        )

        return success_response(
            history,
            "Analysis history retrieved."
        )

    except ValueError as error:

        return error_response(
            str(error),
            400
        )

    except Exception:

        log_error(
            "Analysis history retrieval failed."
        )

        return error_response(
            "Unable to retrieve analysis history.",
            500,
            "HISTORY_ERROR"
        )


# ======================================
# HEALTH CHECK
# ======================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return success_response(
        {
            "status": "healthy",
            "application": APP_NAME,
            "version": APP_VERSION
        },
        "Application is running."
    )


# ======================================
# REQUEST SIZE ERROR
# ======================================

@app.errorhandler(413)
def request_too_large(error):

    return error_response(
        "Uploaded request exceeds the permitted size.",
        413,
        "REQUEST_TOO_LARGE"
    )


# ======================================
# PAGE NOT FOUND
# ======================================

@app.errorhandler(404)
def page_not_found(error):

    if request.path.startswith("/api/"):

        return error_response(
            "Requested API endpoint was not found.",
            404,
            "NOT_FOUND"
        )

    return error_response(
        "Requested page was not found.",
        404,
        "NOT_FOUND"
    )


# ======================================
# INTERNAL SERVER ERROR
# ======================================

@app.errorhandler(500)
def internal_server_error(error):

    log_error(
        "Internal server error."
    )

    return error_response(
        "An internal server error occurred.",
        500,
        "INTERNAL_SERVER_ERROR"
    )


# ======================================
# APPLICATION STARTUP
# ======================================

if __name__ == "__main__":

    log_info(
        f"{APP_NAME} version {APP_VERSION} starting."
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=DEBUG
    )
