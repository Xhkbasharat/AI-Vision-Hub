
# 🚀 AI VISION HUB — DEPLOYMENT GUIDE

### Deploy. Scale. Innovate.

This guide explains how to install, configure, and deploy AI Vision Hub in local and cloud environments.

---

## 📋 DEPLOYMENT REQUIREMENTS

Before deployment, prepare:

- Python 3.11
- Git
- Docker Desktop or Docker Engine
- A GitHub account
- A Render account for cloud deployment

AI Vision Hub uses TensorFlow, PyTorch, OpenCV, and other AI libraries. Deployment can require significant memory, storage, and startup time.

---

## 🖥️ OPTION 1 — LOCAL PYTHON DEPLOYMENT

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Vision-Hub.git
```

### Step 2: Enter the project directory

```bash
cd AI-Vision-Hub
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate it.

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure environment variables

Copy `.env.example` to `.env`.

```bash
cp .env.example .env
```

On Windows:

```bash
copy .env.example .env
```

Generate a secure secret key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Add the generated value to `.env`.

### Step 6: Start the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🐳 OPTION 2 — DOCKER DEPLOYMENT

### Step 1: Build the Docker image

```bash
docker build -t ai-vision-hub:latest .
```

### Step 2: Start the container

```bash
docker run -d \
  --name ai-vision-hub \
  -p 5000:5000 \
  ai-vision-hub:latest
```

### Step 3: Verify the container

```bash
docker ps
```

### Step 4: Check application logs

```bash
docker logs ai-vision-hub
```

### Step 5: Stop the container

```bash
docker stop ai-vision-hub
```

---

## ⚙️ OPTION 3 — DOCKER COMPOSE

Docker Compose simplifies application startup and persistent storage configuration.

### Start the application

```bash
docker compose up -d --build
```

### View running services

```bash
docker compose ps
```

### View logs

```bash
docker compose logs -f
```

### Stop the application

```bash
docker compose down
```

Named Docker volumes preserve data when containers are recreated. They are not a substitute for backups.

---

## ☁️ OPTION 4 — RENDER CLOUD DEPLOYMENT

AI Vision Hub includes a `render.yaml` deployment blueprint.

### Step 1: Prepare GitHub

Push the latest project files to your GitHub repository.

### Step 2: Open Render

Visit:

https://render.com/

Sign in and connect your GitHub account.

### Step 3: Create a Blueprint

1. Select New.
2. Select Blueprint.
3. Choose your AI Vision Hub repository.
4. Allow Render to detect `render.yaml`.

### Step 4: Review deployment settings

Check:

- Dockerfile location
- Python application configuration
- Environment variables
- Health check endpoint
- Persistent disk configuration
- Hosting plan and costs

### Step 5: Deploy

Start the deployment and monitor the build logs.

### Step 6: Verify the application

Open the Render-provided application URL.

Check:

```text
/health
```

The endpoint should return a successful health response.

---

## 🔐 ENVIRONMENT VARIABLES

| Variable | Purpose |
|---|---|
| APP_ENV | Application environment |
| SECRET_KEY | Flask session security |
| DEBUG | Development debugging |
| MAX_CONTENT_LENGTH | Maximum upload size |
| FACE_MATCH_THRESHOLD | Face matching threshold |
| OBJECT_CONFIDENCE_THRESHOLD | Object detection threshold |
| SENTIMENT_MODEL_NAME | Sentiment model |
| CHATBOT_MODEL_NAME | Chatbot model |
| COOKIE_SECURE | Secure session cookie setting |

Never commit the actual `.env` file.

For public deployment, use production settings and HTTPS.

---

## 💾 PERSISTENT STORAGE

AI Vision Hub stores application data in these locations:

| Directory | Purpose |
|---|---|
| `data/uploads/` | Uploaded images |
| `data/known_faces/` | Registered face images |
| `models/` | AI model artifacts |
| `logs/` | Application logs |

Ensure that persistent storage is configured for every directory containing data that must survive redeployment.

Back up databases and biometric data securely. Restrict access to registered face images and model files.

---

## 🧠 MODEL DEPLOYMENT

Some AI modules require pretrained or locally trained models.

| Feature | Requirement |
|---|---|
| Face Recognition | Trained LBPH model and face labels |
| Object Detection | YOLO model weights |
| Image Classification | MobileNetV2 weights |
| Sentiment Analysis | Hugging Face model |
| Fake News Detection | Trained joblib classifier |
| Chatbot | DialoGPT model |

Model files must be available in their configured locations and compatible with the installed libraries.

The fake news feature requires a trained model before it can be used.

---

## 🩺 HEALTH CHECK

AI Vision Hub provides:

```text
GET /health
```

Use this endpoint to verify whether the Flask application responds.

A successful response confirms application availability, not that every AI model is loaded or operational.

---

## 🛡️ PRODUCTION SECURITY CHECKLIST

Before exposing the application publicly:

- [ ] Set `APP_ENV=production`.
- [ ] Disable debug mode.
- [ ] Configure a strong secret key.
- [ ] Enable HTTPS.
- [ ] Enable secure session cookies.
- [ ] Implement authentication and authorization.
- [ ] Implement CSRF protection.
- [ ] Add API rate limiting.
- [ ] Protect biometric data.
- [ ] Configure database backups.
- [ ] Configure persistent model storage.
- [ ] Review dependency vulnerabilities.
- [ ] Verify all AI modules individually.

---

## 🔧 TROUBLESHOOTING

### Application does not start

Check the application logs and verify dependencies.

### Model loading fails

Confirm model files and configured model names.

### Database records disappear

Check that the database directory is backed by persistent storage.

### Image upload fails

Check the file extension, image format, file size, and upload directory permissions.

### Deployment times out

Large AI libraries and model downloads can increase startup time. Check available memory, disk capacity, and deployment logs.

---

## ⚡ FINAL NOTE

AI Vision Hub is under active development. Verify all modules and security controls before production deployment.

<p align="center">
  <strong>AI VISION HUB</strong>
  <br />
  <em>Intelligence. Vision. Innovation.</em>
</p>
