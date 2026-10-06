"""FastAPI application entry point for the AI Image Detector API."""

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ai_image_detector")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "svm_v2.pkl"

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

IMAGE_SIZE = (224, 224)

FEATURE_DIM = 768

AI_THRESHOLD = 0.5


# --------------------------------------------------
# Device
# --------------------------------------------------

def select_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# --------------------------------------------------
# CLIP
# --------------------------------------------------

def load_clip(
    device: torch.device,
) -> tuple[CLIPModel, CLIPProcessor]:
    """Load CLIP model and processor."""

    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        CLIP_MODEL_NAME
    )

    model.to(device)

    model.eval()

    return model, processor


# --------------------------------------------------
# SVM
# --------------------------------------------------

def load_svm(path: Path) -> Any:
    """Load the trained V2 SVM."""

    if not path.is_file():
        raise FileNotFoundError(
            f"SVM model not found at '{path}'. "
            "Make sure svm_v2.pkl exists inside models/."
        )

    return joblib.load(path)


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

@torch.no_grad()
def extract_features(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    """
    Extract the 768-dimensional CLIP image feature.

    Pipeline matches the V2 Kaggle pipeline:

    RGB
        ↓
    Resize 224x224
        ↓
    CLIP processor
        ↓
    CLIP get_image_features()
        ↓
    pooler_output
        ↓
    L2 normalization
    """

    # RGB + resize
    image = image.convert("RGB").resize(IMAGE_SIZE)

    # CLIP preprocessing
    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to(device)

    # CLIP feature extraction
    outputs = model.get_image_features(**inputs)

    # Transformers versions can return either
    # a model output object or the tensor directly.
    features = (
        outputs.pooler_output
        if hasattr(outputs, "pooler_output")
        else outputs
    )

    # Safety check
    if features.shape[-1] != FEATURE_DIM:
        raise RuntimeError(
            f"Unexpected CLIP feature dimension "
            f"{features.shape[-1]}; expected {FEATURE_DIM}."
        )

    # IMPORTANT:
    # V2 was trained on L2-normalized CLIP features.
    features = torch.nn.functional.normalize(
        features,
        p=2,
        dim=-1,
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

    # Load CLIP
    model, processor = load_clip(device)

    logger.info(
        "CLIP model loaded: %s",
        CLIP_MODEL_NAME,
    )

    # Load V2 SVM
    svm = load_svm(MODEL_PATH)

    logger.info(
        "V2 SVM loaded: %s",
        MODEL_PATH,
    )

    logger.info(
        "SVM type: %s",
        type(svm),
    )

    logger.info(
        "SVM classes: %s",
        svm.classes_,
    )

    app.state.device = device
    app.state.clip_model = model
    app.state.clip_processor = processor
    app.state.svm = svm

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

        # Extract V2-compatible CLIP features
        features = extract_features(
            image,
            state.clip_model,
            state.clip_processor,
            state.device,
        )

        # V2 SVM probability
        ai_probability = state.svm.predict_proba(
            features
        )[0, 1]

        return float(ai_probability)

    # Run inference without blocking FastAPI
    ai_probability = await run_in_threadpool(
        _infer
    )

    # Classification
    is_ai = ai_probability >= AI_THRESHOLD

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