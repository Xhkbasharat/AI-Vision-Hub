
import uuid
from pathlib import Path

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, jsonify
)
from werkzeug.utils import secure_filename
from PIL import Image

from config import (
    SECRET_KEY,
    UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH,
    ALLOWED_IMAGE_EXTENSIONS,
    MODEL_FOLDER
)

from database import Database

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

database = Database()


# ---------- Utility functions ----------

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


# ---------- Home page ----------

@app.route("/")
def home():

    stats = database.get_dashboard_stats()

    history = database.get_analysis_history(limit=10)

    return render_template(
        "index.html",
        stats=stats,
        history=history
    )


# ---------- Object detection ----------

@app.route("/detect-objects", methods=["POST"])
def detect_objects():

    try:

        from modules.object_detection import ObjectDetection

        image_file = request.files.get("image")
        image_path = save_image(image_file)

        detector = ObjectDetection()

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

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ---------- Image classification ----------

@app.route("/classify-image", methods=["POST"])
def classify_image():

    try:

        from modules.image_classification import ImageClassification

        image_file = request.files.get("image")
        image_path = save_image(image_file)

        classifier = ImageClassification()

        results = classifier.classify_image(image_path)

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

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ---------- Face registration ----------

@app.route("/register-face", methods=["POST"])
def register_face():

    try:

        from modules.face_recognition import FaceRecognition

        name = request.form.get("name", "").strip()

        image_file = request.files.get("image")

        image_path = save_image(image_file)

        recognizer = FaceRecognition()

        saved_path = recognizer.register_face(
            image_path,
            name
        )

        return jsonify({
            "success": True,
            "message": f"Face registered for {name}.",
            "saved_path": saved_path
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ---------- Face model training ----------

@app.route("/train-faces", methods=["POST"])
def train_faces():

    try:

        from modules.face_recognition import FaceRecognition

        recognizer = FaceRecognition()

        count = recognizer.train_model()

        return jsonify({
            "success": True,
            "message": f"Training completed using {count} images."
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ---------- Face recognition ----------

@app.route("/recognize-face", methods=["POST"])
def recognize_face():

    try:

        from modules.face_recognition import FaceRecognition

        image_file = request.files.get("image")
        image_path = save_image(image_file)

        recognizer = FaceRecognition()

        results = recognizer.recognize_face(image_path)

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

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ---------- Sentiment analysis ----------

@app.route("/sentiment", methods=["POST"])
def sentiment():

    try:

        from modules.sentiment_analysis import SentimentAnalysis

        text = request.form.get("text", "")

        analyzer = SentimentAnalysis()

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

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ---------- Fake news classification ----------

@app.route("/fake-news", methods=["POST"])
def fake_news():

    try:

        from modules.fake_news import FakeNewsDetector

        text = request.form.get("text", "")

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

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ---------- Chatbot ----------

@app.route("/chatbot", methods=["POST"])
def chatbot():

    try:

        from modules.chatbot import AIChatbot

        message = request.form.get("message", "")

        # Simple request-based demonstration.
        # Persistent chat sessions will be added later.
        bot = AIChatbot()

        result = bot.chat(message)

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

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ---------- Error handling ----------

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "File exceeds the 10 MB upload limit."
    }), 413


@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "application": "AI Vision Hub"
    })


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
