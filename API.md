
# ⚡ AI VISION HUB — API DOCUMENTATION

### Unified Artificial Intelligence API Reference

AI Vision Hub provides a Flask-based API for computer vision, machine learning, natural language processing, and conversational AI.

---

## 📌 BASE URL

Local development:

```text
http://127.0.0.1:5000
```

Cloud deployment:

```text
https://YOUR-DEPLOYMENT-DOMAIN
```

Replace the cloud URL with your actual deployment address.

---

## 📦 RESPONSE FORMAT

Successful requests generally return:

```json
{
  "success": true,
  "message": "Operation completed.",
  "data": {}
}
```

Failed requests return:

```json
{
  "success": false,
  "error": "An error occurred."
}
```

The contents of `data` depend on the endpoint.

---

# 👁️ 1. COMPUTER VISION APIs

## 1.1 Object Detection

**Endpoint**

```http
POST /detect-objects
```

**Content-Type**

```text
multipart/form-data
```

**Request field**

| Field | Type | Required |
|---|---|---|
| image | Image file | Yes |

**Example using cURL**

```bash
curl -X POST http://127.0.0.1:5000/detect-objects \
  -F "image=@test.jpg"
```

**Response structure**

```json
{
  "success": true,
  "message": "Object detection completed.",
  "data": {
    "objects": [],
    "image": "uploaded-image.jpg"
  }
}
```

The objects array contains the detections returned by the YOLO model.

---

## 1.2 Image Classification

**Endpoint**

```http
POST /classify-image
```

**Content-Type**

```text
multipart/form-data
```

**Request field**

| Field | Type | Required |
|---|---|---|
| image | Image file | Yes |

**Example**

```bash
curl -X POST http://127.0.0.1:5000/classify-image \
  -F "image=@test.jpg"
```

**Response structure**

```json
{
  "success": true,
  "message": "Image classification completed.",
  "data": {
    "success": true,
    "model": "MobileNetV2",
    "predictions": []
  }
}
```

The predictions array contains the model's classification results.

---

## 1.3 Face Registration

**Endpoint**

```http
POST /register-face
```

**Content-Type**

```text
multipart/form-data
```

**Request fields**

| Field | Type | Required |
|---|---|---|
| person_name | String | Yes |
| image | Image file | Yes |

**Example**

```bash
curl -X POST http://127.0.0.1:5000/register-face \
  -F "person_name=Test Person" \
  -F "image=@face.jpg"
```

**Response structure**

```json
{
  "success": true,
  "message": "Face registration completed.",
  "data": {}
}
```

Face registration must be performed only with the person's informed consent.

---

## 1.4 Train Face Recognition Model

**Endpoint**

```http
POST /train-faces
```

No request body is required.

**Example**

```bash
curl -X POST http://127.0.0.1:5000/train-faces
```

**Response structure**

```json
{
  "success": true,
  "message": "Face model training completed.",
  "data": {}
}
```

Training requires a suitable registered face dataset.

---

## 1.5 Face Recognition

**Endpoint**

```http
POST /recognize-face
```

**Content-Type**

```text
multipart/form-data
```

**Request field**

| Field | Type | Required |
|---|---|---|
| image | Image file | Yes |

**Example**

```bash
curl -X POST http://127.0.0.1:5000/recognize-face \
  -F "image=@face.jpg"
```

**Response structure**

```json
{
  "success": true,
  "message": "Face recognition completed.",
  "data": {
    "faces": []
  }
}
```

Recognition results are model estimates, not definitive identity verification.

---

# 🧠 2. MACHINE LEARNING APIs

## 2.1 Machine Learning Prediction

**Endpoint**

```http
POST /ml-prediction
```

**Content-Type**

```text
application/json
```

**Request body**

```json
{
  "values": [5.1, 3.5, 1.4, 0.2],
  "model": "random_forest"
}
```

Supported model names:

- `random_forest`
- `xgboost`

**Example**

```bash
curl -X POST http://127.0.0.1:5000/ml-prediction \
  -H "Content-Type: application/json" \
  -d '{"values":[5.1,3.5,1.4,0.2],"model":"random_forest"}'
```

**Response structure**

```json
{
  "success": true,
  "message": "Machine learning prediction completed.",
  "data": {
    "success": true,
    "prediction": "Example prediction",
    "model": "random_forest"
  }
}
```

The model expects four valid Iris feature values.

---

# 💬 3. NATURAL LANGUAGE PROCESSING APIs

## 3.1 Sentiment Analysis

**Endpoint**

```http
POST /sentiment
```

**Content-Type**

```text
application/json
```

**Request body**

```json
{
  "text": "This application is amazing!"
}
```

**Example**

```bash
curl -X POST http://127.0.0.1:5000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"This application is amazing!"}'
```

**Response structure**

```json
{
  "success": true,
  "message": "Sentiment analysis completed.",
  "data": {
    "success": true,
    "sentiment": "POSITIVE",
    "confidence": 0.98
  }
}
```

The exact sentiment and confidence values depend on model output.

---

## 3.2 Fake News Detection

**Endpoint**

```http
POST /fake-news
```

**Content-Type**

```text
application/json
```

**Request body**

```json
{
  "text": "Example news article text goes here."
}
```

**Example**

```bash
curl -X POST http://127.0.0.1:5000/fake-news \
  -H "Content-Type: application/json" \
  -d '{"text":"Example news article text goes here."}'
```

**Response structure**

```json
{
  "success": true,
  "message": "Fake news analysis completed.",
  "data": {
    "success": true,
    "prediction": "Real",
    "confidence": 0.85
  }
}
```

The classifier requires a trained model. Its output is a statistical classification, not independent verification of a news claim.

---

## 3.3 AI Chatbot

**Endpoint**

```http
POST /chatbot
```

**Content-Type**

```text
application/json
```

**Request body**

```json
{
  "message": "What is artificial intelligence?"
}
```

**Example**

```bash
curl -X POST http://127.0.0.1:5000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message":"What is artificial intelligence?"}'
```

**Response structure**

```json
{
  "success": true,
  "message": "Chatbot response generated.",
  "data": {
    "response": "Artificial intelligence is...",
    "history": [],
    "model": "DialoGPT"
  }
}
```

Chat history is maintained through the Flask session cookie.

---

## 3.4 Reset Chat History

**Endpoint**

```http
POST /reset-chat
```

**Example**

```bash
curl -X POST http://127.0.0.1:5000/reset-chat
```

**Response**

```json
{
  "success": true,
  "message": "Chat history cleared.",
  "data": null
}
```

---

# 📊 4. DATABASE AND SYSTEM APIs

## 4.1 Analysis History

**Endpoint**

```http
GET /analysis-history
```

**Query parameter**

| Parameter | Type | Default | Allowed |
|---|---|---|---|
| limit | Integer | 50 | 1–500 |

**Example**

```bash
curl "http://127.0.0.1:5000/analysis-history?limit=20"
```

**Response structure**

```json
{
  "success": true,
  "message": "Analysis history retrieved.",
  "data": {
    "history": [],
    "count": 0
  }
}
```

---

## 4.2 Application Health

**Endpoint**

```http
GET /health
```

**Example**

```bash
curl http://127.0.0.1:5000/health
```

**Response**

```json
{
  "success": true,
  "message": "Application is healthy.",
  "data": {
    "app_name": "AI Vision Hub",
    "version": "1.0.0",
    "status": "running"
  }
}
```

This checks application availability. It does not confirm that every AI model is loaded and operational.

---

# ⚠️ 5. HTTP STATUS CODES

| Status | Meaning |
|---|---|
| 200 | Request successful |
| 400 | Invalid input or request |
| 404 | Endpoint not found |
| 405 | HTTP method not allowed |
| 413 | Uploaded file too large |
| 503 | Required fake-news model unavailable |
| 500 | Internal server error |

---

# 🔒 6. API SECURITY

Before exposing APIs publicly, implement:

- Authentication and authorization.
- CSRF protection for browser-based requests.
- API rate limiting.
- HTTPS.
- Secure session cookie configuration.
- Input and upload validation.
- Access restrictions for biometric information.
- Appropriate logging and monitoring.

Never send private face images, passwords, or secret keys in public issue reports.

---

# 🧪 7. API TESTING

Check application health:

```bash
curl http://127.0.0.1:5000/health
```

Run the project tests:

```bash
python run_tests.py
```

---

# 📌 8. IMPORTANT NOTES

- API responses reflect the actual outputs of the configured models.
- Example response values are illustrative.
- Face recognition requires consent and appropriate access controls.
- Fake news detection requires a trained classifier.
- Large AI models may increase processing time and memory usage.
- The current API does not yet implement authentication or CSRF protection.

---

<p align="center">
  <strong>⚡ AI VISION HUB ⚡</strong>
  <br />
  <em>One Platform. Multiple AI Capabilities.</em>
</p>
