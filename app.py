
import os
import uuid
import threading

from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session
)

from werkzeug.utils import secure_filename
from PIL import Image

from config import (
    SECRET_KEY,
    UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH,
    ALLOWED_IMAGE_EXTENSIONS
)

from database import Database


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)

app.secret_key = SECRET_KEY

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

database = Database()


# ==========================================
# AI MODEL INSTANCES
# ==========================================

chatbot_instance = None
sentiment_instance = None
classification_instance = None
object_detection_instance = None
face_recognition_instance = None
ml_prediction_instance = None

model_lock = threading.Lock()


def get_model(model_name):

    global chatbot_instance
    global sentiment_instance
    global classification_instance
    global object_detection_instance
    global face_recognition_instance
    global ml_prediction_instance

    with model_lock:

        if model_name == "chatbot":

            if chatbot_instance is None:
                from modules.chatbot import AIChatbot
                chatbot_instance = AIChatbot()

            return chatbot_instance

        if model_name == "sentiment":

            if sentiment_instance is None:
                from modules.sentiment_analysis import SentimentAnalysis
                sentiment_instance = SentimentAnalysis()

            return sentiment_instance

        if model_name == "classification":

            if classification_instance is None:
                from modules.image_classification import ImageClassification
                classification_instance = ImageClassification()

            return classification_instance

        if model_name == "object_detection":

            if object_detection_instance is None:
                from modules.object_detection import ObjectDetection
                object_detection_instance = ObjectDetection()

            return object_detection_instance

        if model_name == "face":

            if face_recognition_instance is None:
                from modules.face_recognition import FaceRecognition
                face_recognition_instance = FaceRecognition()

            return face_recognition_instance

        if model_name == "ml":

            if ml_prediction_instance is None:
                from modules.ml_prediction import MLPrediction
                ml_prediction_instance = MLPrediction()

            return ml_prediction_instance

        raise ValueError("Unknown AI model.")


# ==========================================
# FILE UPLOAD UTILITIES
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_IMAGE_EXTENSIONS
    )


def save_image(file):

    if not file or not file.filename:
        raise ValueError("Please select an image.")

    if not allowed_file(file.filename):
        raise ValueError("Unsupported image format.")

    extension = file.filename.rsplit(".", 1)[1].lower()

    filename = secure_filename(
        f"{uuid.uuid4().hex}.{extension}"
    )

    file_path = Path(UPLOAD_FOLDER) / filename

    file.stream.seek(0)

    with Image.open(file.stream) as img:

        img.verify()

    file.stream.seek(0)

    file.save(file_path)

    database.add_upload(filename, extension)

    return file_path


# ==========================================
# HOME DASHBOARD
# ==========================================

@app.route("/")
def home():

    stats = database.get_dashboard_stats()

    history = database.get_analysis_history(limit=10)

    return render_template(
        "index.html",
        stats=stats,
        history=history
    )


# ==========================================
# OBJECT DETECTION
# ==========================================

@app.route("/detect-objects", methods=["POST"])
def detect_objects():

    image_path = None

    try:

        from modules.object_detection import ObjectDetection

        image_path = save_image(
            request.files.get("image")
        )

        detector = get_model("object_detection")

        results = detector.detect_objects(image_path)

        database.save_analysis(
            "Object Detection",
            image_path.name,
            str(results)
        )

        return jsonify({
            "success": True,
            "results": results
        })

    except Exception as error:

        app.logger.exception("Object detection failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


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
            str(image_path)
        )

        database.save_analysis(
            "Image Classification",
            image_path.name,
            str(results)
        )

        return jsonify({
            "success": True,
            "results": results
        })

    except Exception as error:

        app.logger.exception("Image classification failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ==========================================
# FACE REGISTRATION
# ==========================================

@app.route("/register-face", methods=["POST"])
def register_face():

    try:

        name = request.form.get(
            "name", ""
        ).strip()

        if not name:
            raise ValueError("Person name is required.")

        image_path = save_image(
            request.files.get("image")
        )

        recognizer = get_model("face")

        saved_path = recognizer.register_face(
            image_path,
            name
        )

        return jsonify({
            "success": True,
            "message": "Face registered successfully.",
            "saved_path": str(saved_path)
        })

    except Exception as error:

        app.logger.exception("Face registration failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ==========================================
# FACE MODEL TRAINING
# ==========================================

@app.route("/train-faces", methods=["POST"])
def train_faces():

    try:

        recognizer = get_model("face")

        count = recognizer.train_model()

        return jsonify({
            "success": True,
            "message": "Face model trained successfully.",
            "training_images": count
        })

    except Exception as error:

        app.logger.exception("Face training failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ==========================================
# FACE RECOGNITION
# ==========================================

@app.route("/recognize-face", methods=["POST"])
def recognize_face():

    try:

        image_path = save_image(
            request.files.get("image")
        )

        recognizer = get_model("face")

        results = recognizer.recognize_face(
            image_path
        )

        database.save_analysis(
            "Face Recognition",
            image_path.name,
            str(results)
        )

        return jsonify({
            "success": True,
            "results": results
        })

    except Exception as error:

        app.logger.exception("Face recognition failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

@app.route("/sentiment", methods=["POST"])
def sentiment():

    try:

        text = request.form.get(
            "text", ""
        ).strip()

        analyzer = get_model("sentiment")

        result = analyzer.analyze_sentiment(text)

        database.save_analysis(
            "Sentiment Analysis",
            text,
            str(result),
            result["confidence"]
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as error:

        app.logger.exception("Sentiment analysis failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ==========================================
# FAKE NEWS CLASSIFICATION
# ==========================================

@app.route("/fake-news", methods=["POST"])
def fake_news():

    try:

        from modules.fake_news import FakeNewsDetector

        text = request.form.get(
            "text", ""
        ).strip()

        detector = FakeNewsDetector()

        result = detector.predict(text)

        database.save_analysis(
            "Fake News Classification",
            text,
            str(result)
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as error:

        app.logger.exception("Fake news classification failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ==========================================
# MACHINE LEARNING PREDICTION
# ==========================================

@app.route("/ml-prediction", methods=["POST"])
def ml_prediction():

    try:

        model = get_model("ml")

        values = request.get_json(silent=True)

        if not isinstance(values, dict):
            raise ValueError("Please provide JSON input.")

        features = values.get("features")
        model_name = values.get(
            "model", "random_forest"
        )

        if not isinstance(features, list):
            raise ValueError(
                "Features must be provided as a list."
            )

        result = model.predict(
            features,
            model_name=model_name
        )

        database.save_analysis(
            "ML Prediction",
            str(features),
            str(result)
        )

        return jsonify({
            "success": True,
            "result": result,
            "model_accuracy": model.get_model_accuracy()
        })

    except Exception as error:

        app.logger.exception("ML prediction failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ==========================================
# AI CHATBOT
# ==========================================

@app.route("/chatbot", methods=["POST"])
def chatbot():

    try:

        message = request.form.get(
            "message", ""
        ).strip()

        if not message:
            raise ValueError("Please enter a message.")

        bot = get_model("chatbot")

        history = session.get(
            "chat_history", []
        )

        result = bot.chat(
            message,
            history=history
        )

        session["chat_history"] = (
            result["history"][-12:]
        )

        database.save_analysis(
            "AI Chatbot",
            message,
            result["bot_response"]
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as error:

        app.logger.exception("Chatbot failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ==========================================
# RESET CHAT
# ==========================================

@app.route("/reset-chat", methods=["POST"])
def reset_chat():

    session.pop("chat_history", None)

    return jsonify({
        "success": True,
        "message": "Chat history cleared."
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
# ERROR HANDLING
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "Image exceeds the 10 MB limit."
    }), 413


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
