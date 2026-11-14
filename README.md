
# AI Vision Hub

An integrated Artificial Intelligence and Machine Learning platform combining computer vision, deep learning, natural language processing, and classical machine learning in one Flask application.

## Overview

AI Vision Hub brings multiple AI capabilities together through a single dashboard.

The project is designed as an educational and development platform for experimenting with different AI technologies.

## Features

### Computer Vision
- Face detection and recognition using OpenCV.
- Face registration and model training.
- Object detection using YOLO.
- Image classification using MobileNetV2.

### Machine Learning
- Iris flower classification.
- Random Forest classifier.
- XGBoost classifier.

### Natural Language Processing
- Sentiment analysis using Hugging Face Transformers.
- Fake-news text classification.
- Conversational chatbot using DialoGPT.

### Dashboard and Backend
- Flask web application.
- SQLite database.
- Image upload handling.
- Analysis history storage.
- Lazy loading of AI models.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Web application |
| OpenCV | Computer vision |
| MediaPipe | Vision processing dependency |
| Ultralytics YOLO | Object detection |
| TensorFlow | Deep learning |
| MobileNetV2 | Image classification |
| PyTorch | Deep learning |
| Hugging Face Transformers | NLP and chatbot |
| Scikit-learn | Machine learning |
| XGBoost | Gradient boosting |
| SQLite | Database |
| NumPy and Pandas | Data processing |

## Project Structure

```text
AI-Vision-Hub/
│
├── app.py
├── config.py
├── database.py
├── requirements.txt
├── train_fake_news.py
├── README.md
├── .gitignore
│
├── modules/
│   ├── __init__.py
│   ├── face_recognition.py
│   ├── object_detection.py
│   ├── image_classification.py
│   ├── ml_prediction.py
│   ├── sentiment_analysis.py
│   ├── fake_news.py
│   └── chatbot.py
│
├── models/
│   └── .gitkeep
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── data/
    ├── uploads/
    ├── known_faces/
    └── fake_news_dataset.csv
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Vision-Hub.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Open the project folder

```bash
cd AI-Vision-Hub
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Some AI packages require compatible Python versions and platform-specific installations. Python 3.10 or 3.11 is a sensible starting point, but verify compatibility with the pinned package versions.

### 5. Configure the secret key

Set a strong secret key before running the application.

Linux or macOS:

```bash
export SECRET_KEY="replace-with-a-long-random-secret"
```

Windows PowerShell:

```powershell
$env:SECRET_KEY="replace-with-a-long-random-secret"
```

Do not publish your real secret key.

### 6. Run the application

```bash
python app.py
```

Open this address in your browser:

```text
http://127.0.0.1:5000
```

## Fake News Model Training

The fake-news classifier requires a labelled dataset.

Create:

```text
data/fake_news_dataset.csv
```

The CSV must contain these columns:

```csv
text,label
```

Use a properly sourced and labelled dataset containing both `REAL` and `FAKE` labels.

After adding the dataset, run:

```bash
python train_fake_news.py
```

The trained model will be saved under:

```text
models/fake_news_model.joblib
```

The fake-news classifier identifies learned text patterns. It does not independently verify claims or establish whether an article is true.

## Available API Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard |
| `/detect-objects` | POST | Object detection |
| `/classify-image` | POST | Image classification |
| `/register-face` | POST | Register a face |
| `/train-faces` | POST | Train face recognizer |
| `/recognize-face` | POST | Recognize a face |
| `/sentiment` | POST | Sentiment analysis |
| `/fake-news` | POST | Fake-news classification |
| `/ml-prediction` | POST | Iris prediction |
| `/chatbot` | POST | Chatbot |
| `/reset-chat` | POST | Reset chatbot history |
| `/health` | GET | Application health |

## Example ML Prediction Request

```json
{
  "values": [5.1, 3.5, 1.4, 0.2],
  "model": "random_forest"
}
```

Supported models:

- `random_forest`
- `xgboost`

## Important Notes

- AI models may download pretrained weights the first time they are used.
- Some models require substantial RAM, disk space, and processing power.
- GPU acceleration depends on compatible hardware and software.
- Do not commit private images, biometric data, local databases, or model weights.
- Face recognition requires informed consent and secure handling of biometric information.
- This project is an educational demonstration and is not intended to make high-stakes decisions.

## Future Improvements

- Authentication and user roles.
- Secure biometric data management.
- Improved model evaluation.
- Real-time camera processing.
- More NLP tools.
- Interactive analytics dashboard.
- API documentation.
- Automated testing and deployment.

## License

Choose and add an appropriate open-source license before distributing this project.

---

**AI Vision Hub — One Platform, Multiple AI Capabilities.**
