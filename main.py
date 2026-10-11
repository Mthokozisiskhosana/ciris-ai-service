"""
CIRIS AI Service
Wraps the trained photo classifier and the duplicate-detection model behind
HTTP endpoints the Node backend can call. Run this on its own, separate from
server.js, matching the architecture in the team handbook (Section 9.7).

Model files (produced by the training notebook, Cirsi_fixed.ipynb — place
them in this folder):
    classifier.pkl            scikit-learn classifier, answers with category names
    duplicate_model.pkl       scikit-learn duplicate detector, works on pairs of photos
    feature_extractor.onnx    EfficientNet-B0 feature extractor, used by both models
    model_info.json           categories, thresholds, preprocessing

Run with:
    uvicorn main:app --reload --port 8000
"""

import io
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import onnxruntime as ort
import sklearn
from PIL import Image, ImageOps

HERE = Path(__file__).parent

app = FastAPI(title="CIRIS AI Service")

# Allow the Node backend (running on a different port) to call this service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual Node server's origin before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained models once, at startup — not on every request.
model = None
feature_session = None
model_info = {}
try:
    model_info = json.loads((HERE / "model_info.json").read_text())
    model = joblib.load(HERE / "classifier.pkl")
    feature_session = ort.InferenceSession(str(HERE / "feature_extractor.onnx"), providers=["CPUExecutionProvider"])
    print("Classifier loaded successfully.")
    print("Categories:", list(model.classes_))
    print("Confidence threshold:", model_info.get("confidence_threshold"))
    trained_with = model_info.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        print(f"WARNING: model was trained with scikit-learn {trained_with} but {sklearn.__version__} "
              f"is installed. Pin scikit-learn=={trained_with} in requirements.txt.")
except FileNotFoundError as err:
    model = feature_session = None
    print(f"WARNING: classifier files missing ({err.filename}). "
          "Place classifier.pkl, feature_extractor.onnx and model_info.json in this folder.")

# Features per pair of photos: |a-b|, a*b (1280 each) and their cosine similarity.
PAIR_FEATURES = 2 * 1280 + 1

dup_model = None
try:
    dup_model = joblib.load(HERE / "duplicate_model.pkl")
    if getattr(dup_model, "n_features_in_", None) != PAIR_FEATURES:
        dup_model = None
        print("WARNING: duplicate_model.pkl is the old model (different features). "
              "Replace it with the one from the training notebook.")
    elif feature_session is None or "duplicate" not in model_info:
        dup_model = None
        print("WARNING: duplicate detection needs feature_extractor.onnx and model_info.json too.")
    else:
        print("Duplicate detection model loaded successfully.")
        print("Duplicate threshold:", model_info["duplicate"]["threshold"])
except FileNotFoundError:
    print("WARNING: duplicate_model.pkl not found. Place it in this same folder.")


# ======================================
# CLASSIFIER FEATURES
# Must match preprocess() in the training notebook exactly, or the model
# gets features it was never trained on and its answers are meaningless.
# ======================================
def preprocess(img: Image.Image) -> np.ndarray:
    size = model_info.get("image_size", 224)
    mean = np.array(model_info.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(model_info.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

    img = img.resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - mean) / std
    return x.transpose(2, 0, 1)


def classifier_features(image_bytes: bytes):
    """EfficientNet-B0 features (1, 1280) for one photo, or None if unreadable."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Same loading as training: respect the phone's rotation flag, then RGB.
            rgb = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return None
    batch = preprocess(rgb)[np.newaxis]
    return feature_session.run(None, {"input": batch})[0]


# ======================================
# DUPLICATE-DETECTION FEATURES
# Must match pair_features() in the training notebook exactly.
# ======================================
def pair_features(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    f1 = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)
    f2 = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)
    cosine = (f1 * f2).sum(axis=1, keepdims=True)
    return np.hstack([np.abs(f1 - f2), f1 * f2, cosine])


@app.get("/")
def health_check():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "categories": [str(c) for c in model.classes_] if model is not None else [],
    }


@app.post("/classify")
async def classify(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check classifier.pkl, feature_extractor.onnx and model_info.json are present.")

    features = classifier_features(await image.read())
    if features is None:
        raise HTTPException(status_code=400, detail="Could not read image. Ensure it's a valid JPEG/PNG.")

    probabilities = model.predict_proba(features)[0]
    best = int(np.argmax(probabilities))
    confidence = float(probabilities[best])
    top_category = str(model.classes_[best])

    # Below the threshold chosen in training, don't guess: the backend
    # stores no category and staff see "Unclassified".
    threshold = float(model_info.get("confidence_threshold", 0.5))
    category = top_category if confidence >= threshold else None

    return {
        "category": category,
        "confidence": round(confidence, 3),
        "top_category": top_category,
    }


@app.post("/check-duplicate")
async def check_duplicate(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    if dup_model is None:
        raise HTTPException(status_code=503, detail="Duplicate detection model not loaded. Check duplicate_model.pkl, feature_extractor.onnx and model_info.json are present.")

    bytes1 = await image1.read()
    bytes2 = await image2.read()

    features1 = classifier_features(bytes1)
    features2 = classifier_features(bytes2)

    if features1 is None or features2 is None:
        raise HTTPException(status_code=400, detail="Could not read one of the images.")

    # Probability that both photos show the same problem. Only at or above the
    # threshold chosen in training is it called a duplicate (the backend then
    # rejects the new report, so the threshold favours precision).
    p_duplicate = float(dup_model.predict_proba(pair_features(features1, features2))[0, 1])
    is_duplicate = p_duplicate >= float(model_info["duplicate"]["threshold"])
    # Confidence is the model's probability for whichever answer it gave.
    confidence = p_duplicate if is_duplicate else 1 - p_duplicate

    return {
        "is_duplicate": is_duplicate,
        "confidence": round(confidence, 3)
    }
