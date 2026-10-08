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
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
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

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Module-level reference to the Telegram Application
telegram_app: Application | None = None


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
# Shared inference helper
# --------------------------------------------------

def _run_inference(image: Image.Image, state: Any) -> dict:
    """Run the full inference pipeline on a PIL Image.

    Returns a dict with 'prediction', 'ai_probability', 'confidence'.
    This is the single source of truth used by both the REST endpoint
    and the Telegram bot handler.
    """

    # Extract DINOv3 features
    features = extract_features(
        image,
        state.dinov3_model,
        state.dinov3_processor,
        state.device,
    )

    # Scale features
    features_scaled = state.scaler.transform(features)

    # Classifier probability
    ai_probability = float(
        state.clf.predict_proba(features_scaled)[0, 1]
    )

    # Classification using trained threshold
    is_ai = ai_probability >= state.threshold

    prediction = "AI Generated" if is_ai else "Real"

    confidence = (
        ai_probability if is_ai else 1.0 - ai_probability
    )

    return {
        "prediction": prediction,
        "ai_probability": ai_probability,
        "confidence": confidence,
    }


# --------------------------------------------------
# Telegram bot handlers
# --------------------------------------------------

async def tg_start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle the /start command."""
    await update.message.reply_text(
        "👋 Send me an image and I'll tell you whether "
        "it is AI-generated or real."
    )


async def tg_handle_image(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle incoming photos: run inference and reply."""

    message = update.message

    # Get the highest-resolution version of the photo
    photo = message.photo[-1]

    telegram_file = await context.bot.get_file(
        photo.file_id
    )

    # Download image into memory
    image_bytes = await telegram_file.download_as_bytearray()

    await message.reply_text("🔍 Analyzing image...")

    try:
        # Validate the image (same checks as /predict)
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ):
            await message.reply_text(
                "❌ That doesn't look like a valid image."
            )
            return

        state = app.state

        # Run inference in a thread to avoid blocking
        result = await run_in_threadpool(
            _run_inference, image, state
        )

        confidence_pct = result["confidence"] * 100

        await message.reply_text(
            f"🤖 Prediction: {result['prediction']}\n"
            f"🎯 Confidence: {confidence_pct:.2f}%"
        )

    except Exception:
        logger.exception("Telegram image handler error")

        await message.reply_text(
            "❌ Something went wrong while analyzing "
            "the image."
        )


# --------------------------------------------------
# Application lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    global telegram_app

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

    # --------------------------------------------------
    # Start Telegram bot inside the same process
    # --------------------------------------------------

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN environment variable is not set."
        )

    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    telegram_app.add_handler(
        CommandHandler("start", tg_start)
    )
    telegram_app.add_handler(
        MessageHandler(filters.PHOTO, tg_handle_image)
    )

    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling()

    logger.info("Telegram bot started (polling)")

    yield

    # --------------------------------------------------
    # Shutdown Telegram bot
    # --------------------------------------------------

    logger.info("Stopping Telegram bot...")

    await telegram_app.updater.stop()
    await telegram_app.stop()
    await telegram_app.shutdown()

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

    # Run inference without blocking FastAPI
    result = await run_in_threadpool(
        _run_inference, image, state
    )

    return result