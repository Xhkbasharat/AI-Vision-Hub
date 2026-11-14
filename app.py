
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    session,
)

from werkzeug.utils import secure_filename

from PIL import (
    Image,
    UnidentifiedImageError,
)

from config import (
    SECRET_KEY,
    DEBUG,
    MAX_CONTENT_LENGTH,
    ALLOWED_IMAGE_EXTENSIONS,
    UPLOAD_FOLDER,
    COOKIE_SECURE,
    APP_NAME,
    APP_VERSION,
)

from database import Database

from modules.face_recognition import FaceRecognition
from modules.object_detection import ObjectDetection
from modules.image_classification import ImageClassification
from modules.ml_prediction import MLPrediction
from modules.sentiment_analysis import SentimentAnalyzer
from modules.fake_news import FakeNewsDetector
from modules.chatbot import Chatbot


# ==========================================
# FLASK APPLICATION CONFIGURATION
# ==========================================

app = Flask(__name__)

app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = COOKIE_SECURE
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

app.config["APP_NAME"] = APP_NAME
app.config["APP_VERSION"] = APP_VERSION

database = Database()

model_cache = {}


# ==========================================
# RESPONSE HELPERS
# ==========================================

def success_response(data=None, message="Success", status_code=200):

    response = {
        "success": True,
        "message": message,
        "data": data,
    }

    return jsonify(response), status_code


def error_response(message, status_code=400):

    response = {
        "success": False,
        "error": message,
    }

    return jsonify(response), status_code


def record_analysis(
    module_name,
    input_text,
    result,
    confidence=None
):

    try:

        database.save_analysis(
            module_name=module_name,
            input_text=input_text,
            result=result,
            confidence=confidence,
        )

    except Exception as error:

        app.logger.exception(
            "Unable to save analysis history: %s",
            error,
        )


# ==========================================
# MODEL MANAGEMENT
# ==========================================

def get_model(model_name):

    if model_name in model_cache:
        return model_cache[model_name]

    model_classes = {
        "face": FaceRecognition,
        "objects": ObjectDetection,
        "classification": ImageClassification,
        "ml": MLPrediction,
        "sentiment": SentimentAnalyzer,
        "chatbot": Chatbot,
    }

    if model_name not in model_classes:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    try:

        model = model_classes[model_name]()

        model_cache[model_name] = model

        return model

    except Exception as error:

        app.logger.exception(
            "Failed to initialize model %s",
            model_name,
        )

        raise RuntimeError(
            f"Unable to initialize {model_name} model."
        ) from error


# ==========================================
# IMAGE UPLOAD VALIDATION
# ==========================================

def save_image(file):

    if file is None:
        raise ValueError("No image file was provided.")

    if not file.filename:
        raise ValueError("No filename was provided.")

    filename = secure_filename(file.filename)

    if not filename:
        raise ValueError("Invalid filename.")

    extension = Path(filename).suffix.lower().lstrip(".")

    if extension not in ALLOWED_IMAGE_EXTENSIONS:

        raise ValueError(
            "Unsupported image format. "
            "Allowed formats: JPG, JPEG, PNG, WEBP."
        )

    upload_path = None

    try:

        file.stream.seek(0)

        with Image.open(file.stream) as image:

            image_format = image.format

            format_extensions = {
                "JPEG": {"jpg", "jpeg"},
                "PNG": {"png"},
                "WEBP": {"webp"},
            }

            allowed_extensions = format_extensions.get(
                image_format,
                set(),
            )

            if extension not in allowed_extensions:

                raise ValueError(
                    "Image content does not match its file extension."
                )

            width, height = image.size

            if width <= 0 or height <= 0:

                raise ValueError(
                    "Invalid image dimensions."
                )

            if width * height > 25_000_000:

                raise ValueError(
                    "Image dimensions exceed the permitted limit."
                )

            image.verify()

    except ValueError:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as error:

        raise ValueError(
            "The uploaded file is not a valid supported image."
        ) from error

    upload_filename = (
        f"{uuid.uuid4().hex}.{extension}"
    )

    upload_path = Path(UPLOAD_FOLDER) / upload_filename

    try:

        file.stream.seek(0)

        with Image.open(file.stream) as image:

            image.load()

            image.save(
                upload_path,
                format=image.format,
            )

        database.add_upload(
            upload_filename,
            extension,
        )

    except Exception:

        if upload_path.exists():
            upload_path.unlink()

        raise

    return str(upload_path)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def index():

    try:

        stats = database.get_dashboard_stats()

    except Exception:

        app.logger.exception(
            "Unable to retrieve dashboard statistics"
        )

        stats = {
            "total_uploads": 0,
            "registered_faces": 0,
            "total_analyses": 0,
        }

    return render_template(
        "index.html",
        app_name=APP_NAME,
        app_version=APP_VERSION,
        stats=stats,
    )


# ==========================================
# OBJECT DETECTION
# ==========================================

@app.route("/detect-objects", methods=["POST"])
def detect_objects():

    try:

        if "image" not in request.files:

            return error_response(
                "Please upload an image."
            )

        image_path = save_image(
            request.files["image"]
        )

        detector = get_model("objects")

        results = detector.detect_objects(
            image_path
        )

        record_analysis(
            "object_detection",
            Path(image_path).name,
            results,
        )

        return success_response(
            data={
                "objects": results,
                "image": Path(image_path).name,
            },
            message="Object detection completed.",
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        app.logger.exception(
            "Object detection failed"
        )

        return error_response(
            "Object detection failed.",
            500,
        )


# ==========================================
# IMAGE CLASSIFICATION
# ==========================================

@app.route("/classify-image", methods=["POST"])
def classify_image():

    try:

        if "image" not in request.files:

            return error_response(
                "Please upload an image."
            )

        image_path = save_image(
            request.files["image"]
        )

        classifier = get_model("classification")

        result = classifier.classify_image(
            image_path
        )

        record_analysis(
            "image_classification",
            Path(image_path).name,
            result,
        )

        return success_response(
            data=result,
            message="Image classification completed.",
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        app.logger.exception(
            "Image classification failed"
        )

        return error_response(
            "Image classification failed.",
            500,
        )


# ==========================================
# FACE REGISTRATION
# ==========================================

@app.route("/register-face", methods=["POST"])
def register_face():

    try:

        person_name = request.form.get(
            "person_name",
            "",
        ).strip()

        if not person_name:

            return error_response(
                "Person name is required."
            )

        if len(person_name) > 100:

            return error_response(
                "Person name must be 100 characters or fewer."
            )

        if "image" not in request.files:

            return error_response(
                "Please upload a face image."
            )

        image_path = save_image(
            request.files["image"]
        )

        face_model = get_model("face")

        result = face_model.register_face(
            person_name,
            image_path,
        )

        record_analysis(
            "face_registration",
            person_name,
            result,
        )

        return success_response(
            data=result,
            message="Face registration completed.",
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        app.logger.exception(
            "Face registration failed"
        )

        return error_response(
            "Face registration failed.",
            500,
        )


# ==========================================
# TRAIN FACE RECOGNITION MODEL
# ==========================================

@app.route("/train-faces", methods=["POST"])
def train_faces():

    try:

        face_model = get_model("face")

        result = face_model.train_model()

        record_analysis(
            "face_training",
            "Registered face dataset",
            result,
        )

        return success_response(
            data=result,
            message="Face model training completed.",
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        app.logger.exception(
            "Face model training failed"
        )

        return error_response(
            "Face model training failed.",
            500,
        )


# ==========================================
# FACE RECOGNITION
# ==========================================

@app.route("/recognize-face", methods=["POST"])
def recognize_face():

    try:

        if "image" not in request.files:

            return error_response(
                "Please upload an image."
            )

        image_path = save_image(
            request.files["image"]
        )

        face_model = get_model("face")

        results = face_model.recognize_face(
            image_path
        )

        record_analysis(
            "face_recognition",
            Path(image_path).name,
            results,
        )

        return success_response(
            data={
                "faces": results,
            },
            message="Face recognition completed.",
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        app.logger.exception(
            "Face recognition failed"
        )

        return error_response(
            "Face recognition failed.",
            500,
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
            "",
        )

        if not isinstance(text, str):

            return error_response(
                "Text must be a string."
            )

        text = text.strip()

        if not text:

            return error_response(
                "Please enter text for sentiment analysis."
            )

        if len(text) > 5000:

            return error_response(
                "Text exceeds the 5000-character limit."
            )

        analyzer = get_model("sentiment")

        result = analyzer.analyze_sentiment(
            text
        )

        confidence = result.get("confidence")

        record_analysis(
            "sentiment_analysis",
            text,
            result,
            confidence,
        )

        return success_response(
            data=result,
            message="Sentiment analysis completed.",
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        app.logger.exception(
            "Sentiment analysis failed"
        )

        return error_response(
            "Sentiment analysis failed.",
            500,
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
            "",
        )

        if not isinstance(text, str):

            return error_response(
                "Text must be a string."
            )

        text = text.strip()

        if not text:

            return error_response(
                "Please enter text for fake news detection."
            )

        if len(text) > 10000:

            return error_response(
                "Text exceeds the 10000-character limit."
            )

        detector = FakeNewsDetector()

        result = detector.predict(text)

        confidence = result.get("confidence")

        record_analysis(
            "fake_news_detection",
            text,
            result,
            confidence,
        )

        return success_response(
            data=result,
            message="Fake news analysis completed.",
        )

    except ValueError as error:

        return error_response(str(error))

    except FileNotFoundError:

        return error_response(
            "Fake news model is not available. "
            "Train the model before using this feature.",
            503,
        )

    except Exception:

        app.logger.exception(
            "Fake news detection failed"
        )

        return error_response(
            "Fake news detection failed.",
            500,
        )


# ==========================================
# MACHINE LEARNING PREDICTION
# ==========================================

@app.route("/ml-prediction", methods=["POST"])
def ml_prediction():

    try:

        data = request.get_json(silent=True) or {}

        values = data.get(
            "values"
        )

        model_name = data.get(
            "model",
            "random_forest",
        )

        if not isinstance(model_name, str):

            return error_response(
                "Model name must be a string."
            )

        predictor = get_model("ml")

        result = predictor.predict(
            values,
            model_name=model_name,
        )

        record_analysis(
            "ml_prediction",
            str(values),
            result,
            result.get("confidence"),
        )

        return success_response(
            data=result,
            message="Machine learning prediction completed.",
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        app.logger.exception(
            "Machine learning prediction failed"
        )

        return error_response(
            "Machine learning prediction failed.",
            500,
        )


# ==========================================
# CHATBOT
# ==========================================

@app.route("/chatbot", methods=["POST"])
def chatbot():

    try:

        data = request.get_json(silent=True) or {}

        message = data.get(
            "message",
            "",
        )

        if not isinstance(message, str):

            return error_response(
                "Message must be a string."
            )

        message = message.strip()

        if not message:

            return error_response(
                "Please enter a message."
            )

        if len(message) > 2000:

            return error_response(
                "Message exceeds the 2000-character limit."
            )

        history = session.get(
            "chat_history",
            [],
        )

        bot = get_model("chatbot")

        result = bot.chat(
            message,
            history=history,
        )

        session["chat_history"] = result.get(
            "history",
            [],
        )[-8:]

        return success_response(
            data=result,
            message="Chatbot response generated.",
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        app.logger.exception(
            "Chatbot request failed"
        )

        return error_response(
            "Unable to generate chatbot response.",
            500,
        )


# ==========================================
# RESET CHATBOT HISTORY
# ==========================================

@app.route("/reset-chat", methods=["POST"])
def reset_chat():

    session.pop(
        "chat_history",
        None,
    )

    return success_response(
        message="Chat history cleared.",
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
            type=int,
        )

        if limit is None or limit < 1 or limit > 500:

            return error_response(
                "History limit must be between 1 and 500."
            )

        history = database.get_analysis_history(
            limit=limit
        )

        return success_response(
            data={
                "history": history,
                "count": len(history),
            },
            message="Analysis history retrieved.",
        )

    except ValueError as error:

        return error_response(str(error))

    except Exception:

        app.logger.exception(
            "Unable to retrieve analysis history"
        )

        return error_response(
            "Unable to retrieve analysis history.",
            500,
        )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    return success_response(
        data={
            "app_name": APP_NAME,
            "version": APP_VERSION,
            "status": "running",
        },
        message="Application is healthy.",
    )


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return error_response(
        "Uploaded file exceeds the maximum permitted size.",
        413,
    )


@app.errorhandler(404)
def page_not_found(error):

    return error_response(
        "Requested endpoint was not found.",
        404,
    )


@app.errorhandler(405)
def method_not_allowed(error):

    return error_response(
        "HTTP method is not allowed for this endpoint.",
        405,
    )


@app.errorhandler(500)
def internal_server_error(error):

    app.logger.exception(
        "Internal server error"
    )

    return error_response(
        "An internal server error occurred.",
        500,
    )


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=DEBUG,
    )
