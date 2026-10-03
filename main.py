"""
CIRIS AI Service
Wraps the trained Random Forest classifier (classifier.pkl) behind an HTTP
endpoint the Node backend can call. Run this on its own, separate from
server.js, matching the architecture in the team handbook (Section 9.7).

Run with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import joblib
import io

app = FastAPI(title="CIRIS AI Service")

# Allow the Node backend (running on a different port) to call this service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual Node server's origin before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained models once, at startup — not on every request.
try:
    model = joblib.load("classifier.pkl")
    print("Classifier loaded successfully.")
    print("Categories:", list(model.classes_))
except FileNotFoundError:
    model = None
    print("WARNING: classifier.pkl not found. Place it in this same folder.")

try:
    dup_model = joblib.load("duplicate_model.pkl")
    print("Duplicate detection model loaded successfully.")
except FileNotFoundError:
    dup_model = None
    print("WARNING: duplicate_model.pkl not found. Place it in this same folder.")


def extract_features(image_bytes: bytes):
    """
    Identical to the feature extraction used during training in Colab —
    this MUST match exactly, or the model will get features it was never
    trained on and its predictions will be meaningless.
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None

    img = cv2.resize(img, (128, 128))

    color_mean = img.mean(axis=(0, 1))
    color_std = img.std(axis=(0, 1))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean()

    gray_mean = gray.mean()
    gray_std = gray.std()

    return np.concatenate([color_mean, color_std, [edge_density, gray_mean, gray_std]])


@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": model is not None}


@app.post("/classify")
async def classify(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check classifier.pkl is present.")

    image_bytes = await image.read()
    features = extract_features(image_bytes)

    if features is None:
        raise HTTPException(status_code=400, detail="Could not read image. Ensure it's a valid JPEG/PNG.")

    features = features.reshape(1, -1)  # sklearn expects a 2D array, one row per sample

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    return {
        "category": prediction,
        "confidence": round(confidence, 3)
    }


@app.post("/check-duplicate")
async def check_duplicate(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    if dup_model is None:
        raise HTTPException(status_code=503, detail="Duplicate detection model not loaded. Check duplicate_model.pkl is present.")

    bytes1 = await image1.read()
    bytes2 = await image2.read()

    features1 = extract_features(bytes1)
    features2 = extract_features(bytes2)

    if features1 is None or features2 is None:
        raise HTTPException(status_code=400, detail="Could not read one of the images.")

    # Same transformation used during training: absolute difference
    # between the two images' feature vectors.
    diff = np.abs(features1 - features2).reshape(1, -1)

    prediction = dup_model.predict(diff)[0]
    probabilities = dup_model.predict_proba(diff)[0]
    # Class 1 = duplicate, class 0 = not duplicate — confidence is the
    # model's probability for whichever class it predicted.
    confidence = float(probabilities[1] if prediction == 1 else probabilities[0])

    return {
        "is_duplicate": bool(prediction == 1),
        "confidence": round(confidence, 3)
    }