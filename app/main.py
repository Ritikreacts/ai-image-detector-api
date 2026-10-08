"""FastAPI application entry point for the AI Image Detector API."""

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # Load .env before any Hugging Face calls

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ai_image_detector")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "dinov3_linear_clf-v2.pkl"

DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"

IMAGE_SIZE = (224, 224)

FEATURE_DIM = 1024

HF_TOKEN = os.environ.get("HF_TOKEN")


# --------------------------------------------------
# Device
# --------------------------------------------------

def select_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# --------------------------------------------------
# DINOv3
# --------------------------------------------------

def load_dinov3(
    device: torch.device,
) -> tuple[AutoModel, AutoImageProcessor]:
    """Load DINOv3 model and processor."""

    processor = AutoImageProcessor.from_pretrained(
        DINOV3_MODEL_NAME,
        token=HF_TOKEN,
    )

    model = AutoModel.from_pretrained(
        DINOV3_MODEL_NAME,
        token=HF_TOKEN,
    )

    model.to(device)

    model.eval()

    return model, processor


# --------------------------------------------------
# Classifier
# --------------------------------------------------

def load_classifier(path: Path) -> dict:
    """Load the trained DINOv3 linear classifier bundle.

    Returns a dict with keys: 'scaler', 'clf', 'threshold'.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Classifier not found at '{path}'. "
            "Make sure dinov3_linear_clf.pkl exists inside models/."
        )

    return joblib.load(path)


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

@torch.no_grad()
def extract_features(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: torch.device,
) -> np.ndarray:
    """
    Extract the 1024-dimensional DINOv3 image feature.

    Pipeline:

    RGB
        ↓
    Resize 224x224
        ↓
    DINOv3 processor
        ↓
    DINOv3 forward pass
        ↓
    CLS token (last_hidden_state[:, 0])
    """

    # RGB + resize
    image = image.convert("RGB").resize(IMAGE_SIZE)

    # DINOv3 preprocessing
    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to(device)

    # DINOv3 feature extraction
    outputs = model(**inputs)

    # Use the CLS token from the last hidden state
    features = outputs.last_hidden_state[:, 0]

    # Safety check
    if features.shape[-1] != FEATURE_DIM:
        raise RuntimeError(
            f"Unexpected DINOv3 feature dimension "
            f"{features.shape[-1]}; expected {FEATURE_DIM}."
        )

    return features.cpu().numpy()


# --------------------------------------------------
# Application lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    device = select_device()

    logger.info(
        "Selected device: %s",
        device,
    )

    # Load DINOv3
    model, processor = load_dinov3(device)

    logger.info(
        "DINOv3 model loaded: %s",
        DINOV3_MODEL_NAME,
    )

    # Load classifier bundle
    clf_bundle = load_classifier(MODEL_PATH)

    logger.info(
        "Classifier loaded: %s",
        MODEL_PATH,
    )

    logger.info(
        "Classifier type: %s",
        type(clf_bundle["clf"]),
    )

    logger.info(
        "Classifier classes: %s",
        clf_bundle["clf"].classes_,
    )

    logger.info(
        "Threshold: %s",
        clf_bundle["threshold"],
    )

    app.state.device = device
    app.state.dinov3_model = model
    app.state.dinov3_processor = processor
    app.state.scaler = clf_bundle["scaler"]
    app.state.clf = clf_bundle["clf"]
    app.state.threshold = clf_bundle["threshold"]

    yield

    logger.info(
        "Shutting down AI Image Detector API"
    )


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AI Image Detector API",
    description="REST API for detecting AI-generated images.",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "message": "AI Image Detector API is running",
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
) -> dict:

    contents = await file.read()

    # Validate uploaded image
    try:
        image = Image.open(
            io.BytesIO(contents)
        )

        image.load()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )

    state = app.state

    def _infer() -> float:

        # Extract DINOv3 features
        features = extract_features(
            image,
            state.dinov3_model,
            state.dinov3_processor,
            state.device,
        )

        # Scale features
        features_scaled = state.scaler.transform(
            features
        )

        # Classifier probability
        ai_probability = state.clf.predict_proba(
            features_scaled
        )[0, 1]

        return float(ai_probability)

    # Run inference without blocking FastAPI
    ai_probability = await run_in_threadpool(
        _infer
    )

    # Classification using trained threshold
    is_ai = ai_probability >= state.threshold

    prediction = (
        "AI Generated"
        if is_ai
        else "Real"
    )

    confidence = (
        ai_probability
        if is_ai
        else 1.0 - ai_probability
    )

    return {
        "prediction": prediction,
        "ai_probability": ai_probability,
        "confidence": confidence,
    }