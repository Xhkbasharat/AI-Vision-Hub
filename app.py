
import logging
import threading
import uuid
from pathlib import Path

from flask import (
    Flask,
    request,
    jsonify,
    session,
    render_template
)
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from config import (
    SECRET_KEY,
    DEBUG,
    MAX_CONTENT_LENGTH,
    ALLOWED_IMAGE_EXTENSIONS
)

from database import Database

from modules.face_recognition import FaceRecognition
from modules.object_detection import ObjectDetection
from modules.image_classification import ImageClassification
from modules.ml_prediction import MLPrediction
from modules.sentiment_analysis import SentimentAnalysis
from modules.fake_news import FakeNewsDetector
from modules.chatbot import Chatbot


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)

app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

db = Database()

logging.basicConfig(level=logging.INFO)

# Model instances are loaded only when first needed.
model_instances = {}
model_lock = threading.Lock()


# ==========================================
# MODEL LOADER
# ==========================================

def get_model(name):
    with model_lock:
        if name in model_instances:
            return model_instances[name]

        model_classes = {
            "face": FaceRecognition,
            "object": ObjectDetection,
            "classification": ImageClassification,
            "ml": MLPrediction,
            "sentiment": SentimentAnalysis,
            "chatbot": Chatbot
        }

        if name not in model_classes:
            raise ValueError("Unknown AI model.")

        model_instances[name] = model_classes[name]()

        return model_instances[name]


# ==========================================
# IMAGE VALIDATION AND UPLOAD
# ==========================================

def allowed_file(filename):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_IMAGE_EXTENSIONS


def save_image(file):
    if not file or not file.filename:
        raise ValueError("Please select an image.")

    if not allowed_file(file.filename):
        raise ValueError(
            "Unsupported image format. Use JPG, JPEG, PNG or WEBP."
        )

    try:
        image = Image.open(file.stream)
        image.verify()
        file.stream.seek(0)
    except (UnidentifiedImageError, OSError):
        raise ValueError("The uploaded file is not a valid image.")

    original_name = secure_filename(file.filename)

    if not original_name:
        raise ValueError("Invalid filename.")

    extension = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"

    upload_folder = Path("data/uploads")
    upload_folder.mkdir(parents=True, exist_ok=True)

    file_path = upload_folder / unique_name

    file.save(str(file_path))

    db.add_upload(
        filename=unique_name,
        file_type=extension
    )

    return str(file_path)


# ==========================================
# HOME DASHBOARD
# ==========================================

@app.route("/")
def home():
    stats = db.get_dashboard_stats()

    return render_template(
        "index.html",
        stats=stats
    )


# ==========================================
# OBJECT DETECTION
# ==========================================

@app.route("/detect-objects", methods=["POST"])
def detect_objects():
    image_path = None

    try:
        image_path = save_image(request.files.get("image"))

        detector = get_model("object")
        results = detector.detect_objects(image_path)

        db.save_analysis(
            module_name="object_detection",
            input_text=image_path,
            result=str(results),
            confidence=None
        )

        return jsonify({
            "success": True,
            "results": results
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Object detection failed")
        return jsonify({
            "error": "Object detection could not be completed."
        }), 500


# ==========================================
# IMAGE CLASSIFICATION
# ==========================================

@app.route("/classify-image", methods=["POST"])
def classify_image():
    try:
        image_path = save_image(request.files.get("image"))

        classifier = get_model("classification")
        results = classifier.classify_image(image_path)

        db.save_analysis(
            module_name="image_classification",
            input_text=image_path,
            result=str(results),
            confidence=None
        )

        return jsonify({
            "success": True,
            "results": results
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Image classification failed")
        return jsonify({
            "error": "Image classification could not be completed."
        }), 500


# ==========================================
# FACE REGISTRATION
# ==========================================

@app.route("/register-face", methods=["POST"])
def register_face():
    try:
        person_name = request.form.get("person_name", "").strip()

        if not person_name:
            return jsonify({
                "error": "Enter a person's name."
            }), 400

        if len(person_name) > 50:
            return jsonify({
                "error": "Name must be 50 characters or fewer."
            }), 400

        image_path = save_image(request.files.get("image"))

        face_model = get_model("face")

        result = face_model.register_face(
            person_name,
            image_path
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Face registration failed")
        return jsonify({
            "error": "Face registration could not be completed."
        }), 500


# ==========================================
# FACE MODEL TRAINING
# ==========================================

@app.route("/train-faces", methods=["POST"])
def train_faces():
    try:
        face_model = get_model("face")
        result = face_model.train_model()

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception:
        app.logger.exception("Face training failed")
        return jsonify({
            "error": "Face model training failed."
        }), 500


# ==========================================
# FACE RECOGNITION
# ==========================================

@app.route("/recognize-face", methods=["POST"])
def recognize_face():
    try:
        image_path = save_image(request.files.get("image"))

        face_model = get_model("face")
        results = face_model.recognize_face(image_path)

        return jsonify({
            "success": True,
            "results": results
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Face recognition failed")
        return jsonify({
            "error": "Face recognition could not be completed."
        }), 500


# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

@app.route("/sentiment", methods=["POST"])
def sentiment():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")

        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "error": "Enter text to analyze."
            }), 400

        if len(text) > 5000:
            return jsonify({
                "error": "Text must be 5000 characters or fewer."
            }), 400

        analyzer = get_model("sentiment")
        result = analyzer.analyze_sentiment(text)

        db.save_analysis(
            module_name="sentiment_analysis",
            input_text=text,
            result=str(result),
            confidence=result.get("confidence")
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Sentiment analysis failed")
        return jsonify({
            "error": "Sentiment analysis could not be completed."
        }), 500


# ==========================================
# FAKE NEWS DETECTION
# ==========================================

@app.route("/fake-news", methods=["POST"])
def fake_news():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")

        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "error": "Enter a news article or headline."
            }), 400

        if len(text) > 10000:
            return jsonify({
                "error": "Text must be 10000 characters or fewer."
            }), 400

        detector = FakeNewsDetector()
        result = detector.predict(text)

        db.save_analysis(
            module_name="fake_news",
            input_text=text,
            result=str(result),
            confidence=None
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except FileNotFoundError:
        return jsonify({
            "error": (
                "Fake news model is not trained. "
                "Train it using train_fake_news.py first."
            )
        }), 503

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Fake news detection failed")
        return jsonify({
            "error": "Fake news analysis could not be completed."
        }), 500


# ==========================================
# IRIS ML PREDICTION
# ==========================================

@app.route("/ml-prediction", methods=["POST"])
def ml_prediction():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid JSON request."
            }), 400

        values = data.get("values")
        model_name = data.get(
            "model",
            "random_forest"
        )

        if not isinstance(values, list) or len(values) != 4:
            return jsonify({
                "error": "Exactly four measurements are required."
            }), 400

        if model_name not in (
            "random_forest",
            "xgboost"
        ):
            return jsonify({
                "error": "Choose random_forest or xgboost."
            }), 400

        ml_model = get_model("ml")

        result = ml_model.predict(
            values,
            model_name=model_name
        )

        db.save_analysis(
            module_name="ml_prediction",
            input_text=str(values),
            result=result["prediction"],
            confidence=result["confidence"]
        )

        return jsonify(result), 200

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception("ML prediction failed")
        return jsonify({
            "error": "Prediction could not be completed."
        }), 500


# ==========================================
# CHATBOT
# ==========================================

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        message = request.form.get("message", "").strip()

        if not message:
            return jsonify({
                "error": "Enter a message."
            }), 400

        if len(message) > 1000:
            return jsonify({
                "error": "Message must be 1000 characters or fewer."
            }), 400

        bot = get_model("chatbot")

        history = session.get("chat_history", [])

        response = bot.chat(
            message,
            history=history
        )

        session["chat_history"] = response.get(
            "history",
            []
        )[-12:]

        return jsonify({
            "success": True,
            "response": response.get("response", "")
        })

    except Exception:
        app.logger.exception("Chatbot failed")
        return jsonify({
            "error": "Chatbot could not respond."
        }), 500


# ==========================================
# RESET CHAT
# ==========================================

@app.route("/reset-chat", methods=["POST"])
def reset_chat():
    session.pop("chat_history", None)

    return jsonify({
        "success": True,
        "message": "Chat history reset."
    })


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "application": "AI Vision Hub",
        "version": "1.0.0"
    })


# ==========================================
# FILE SIZE ERROR
# ==========================================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Uploaded file is too large."
    }), 413


# ==========================================
# APPLICATION START
# ==========================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=DEBUG
    )
