<p align="center">
  <h1 align="center">🔍 AI Image Detector API</h1>
  <p align="center">
    Detect AI-generated images with a single API call.<br/>
    Powered by <strong>DINOv3</strong> embeddings and a trained linear classifier.
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#telegram-bot">Telegram Bot</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#license">License</a>
</p>

---

## Features

- **REST API** — Upload an image, get back a prediction (`AI Generated` or `Real`) with a confidence score.
- **DINOv3 Backbone** — Uses Facebook's [`dinov3-vitl16`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) vision transformer for robust feature extraction.
- **Lightweight Classifier** — A scikit-learn linear classifier runs on top of DINOv3 embeddings for fast, accurate inference.
- **Telegram Bot** — Send a photo to a Telegram bot and get instant AI detection results.
- **GPU & CPU** — Automatically uses CUDA when available, falls back to CPU.

---

## How It Works

```
Image Upload
    ↓
RGB Conversion + Resize (224×224)
    ↓
DINOv3 ViT-L/16 → CLS Token (1024-d embedding)
    ↓
StandardScaler → Linear Classifier
    ↓
Prediction: "AI Generated" or "Real" + Confidence %
```

The pipeline extracts the CLS token from DINOv3's last hidden state as a 1024-dimensional feature vector. This vector is scaled and passed through a trained linear classifier with an optimized decision threshold.

---

## Quick Start

### Prerequisites

- Python 3.11+
- A [Hugging Face account](https://huggingface.co) with access to the [DINOv3 model](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (gated repo — request access first)
- A Hugging Face API token ([create one here](https://huggingface.co/settings/tokens))

### 1. Clone the repository

```bash
git clone https://github.com/Ritikreacts/ai-image-detector-api.git
cd ai-image-detector-api
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_huggingface_token_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

> **⚠️ Never commit your `.env` file.** It is already listed in `.gitignore`.

### 5. Start the API server

```bash
uvicorn app.main:app --reload
```

The server starts at **http://127.0.0.1:8000**.  
Interactive API docs are available at **http://127.0.0.1:8000/docs**.

> On first launch, the DINOv3 model weights will be downloaded from Hugging Face (~1.2 GB). Subsequent starts use the cached version.

---

## API Reference

### Health Check

```
GET /
```

**Response:**

```json
{
  "status": "ok",
  "message": "AI Image Detector API is running"
}
```

### Predict

```
POST /predict
```

Upload an image file to classify it as AI-generated or real.

**Request:**

| Parameter | Type         | Description          |
|-----------|--------------|----------------------|
| `file`    | `UploadFile` | Image file (required) |

**Example (curl):**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path/to/image.jpg"
```

**Example (PowerShell):**

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -InFile "path\to\image.jpg" `
  -ContentType "multipart/form-data"
```

**Example (Python):**

```python
import httpx

with open("image.jpg", "rb") as f:
    response = httpx.post(
        "http://127.0.0.1:8000/predict",
        files={"file": ("image.jpg", f, "image/jpeg")},
    )

print(response.json())
```

**Response:**

```json
{
  "prediction": "AI Generated",
  "ai_probability": 0.9432,
  "confidence": 0.9432
}
```

| Field             | Type    | Description                                               |
|-------------------|---------|-----------------------------------------------------------|
| `prediction`      | string  | `"AI Generated"` or `"Real"`                              |
| `ai_probability`  | float   | Raw probability that the image is AI-generated (0.0–1.0)  |
| `confidence`      | float   | Confidence in the predicted class (0.0–1.0)               |

---

## Telegram Bot

The Telegram bot runs **inside the same process** as the FastAPI server — no separate script or terminal needed. It starts automatically via long-polling when the server boots.

### Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram and get the bot token.
2. Add the token to your `.env` file (or set it as an environment variable on your hosting platform).
3. Start the server — the bot starts with it.

### Usage

1. Open your bot in Telegram.
2. Send `/start` to see the welcome message.
3. Send any photo — the bot analyzes it directly (no HTTP round-trip) and replies with the prediction and confidence.

---

## Project Structure

```
ai-image-detector-api/
├── app/
│   ├── __init__.py              # Package marker
│   └── main.py                  # FastAPI app + Telegram bot (single process)
├── models/
│   └── dinov3_linear_clf-v2.pkl # Trained linear classifier bundle (scaler + clf + threshold)
├── Dockerfile                   # Production container for Render
├── .dockerignore
├── .env                         # Environment variables (not committed)
├── .gitignore
├── requirements.txt             # Pinned dependencies
└── README.md
```

### Key Components

| File | Role |
|------|------|
| `app/main.py` | FastAPI application with Telegram bot, lifespan startup, DINOv3 loading, feature extraction, and prediction endpoint |
| `models/*.pkl` | Serialized classifier bundle containing a `StandardScaler`, trained classifier, and optimized decision threshold |
| `Dockerfile` | Production container image for deployment on Render (or any Docker-based platform) |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Feature Extraction | [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) via [Transformers](https://huggingface.co/docs/transformers) |
| Deep Learning | [PyTorch](https://pytorch.org/) |
| Classifier | [scikit-learn](https://scikit-learn.org/) |
| Image Processing | [Pillow](https://python-pillow.org/) |
| Telegram Bot | [python-telegram-bot](https://python-telegram-bot.org/) v22.x |
| HTTP Client | [httpx](https://www.python-httpx.org/) |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | ✅ | Hugging Face API token for accessing the gated DINOv3 model |
| `TELEGRAM_BOT_TOKEN` | ✅ | Telegram bot token from [@BotFather](https://t.me/BotFather) |
| `PORT` | No (default: `10000`) | Port for the HTTP server — set automatically by Render |

---

## Security

- **Never commit secrets.** The `.env` file is gitignored.
- The HF token is only used server-side to download model weights.
- The Telegram bot token is only used to authenticate with the Telegram Bot API.
- No user data or images are stored — all inference is done in-memory.

---

## Deployment (Render)

This project is configured for one-click deployment on [Render](https://render.com):

1. Push this repo to GitHub.
2. Create a new **Web Service** on Render, connect the repo.
3. Render auto-detects the `Dockerfile`.
4. Set environment variables in Render's dashboard: `HF_TOKEN` and `TELEGRAM_BOT_TOKEN`.
5. Deploy — the API server and Telegram bot start together as a single process.

The `Dockerfile` respects Render's `$PORT` environment variable automatically.

---

## License

This project is for educational and research purposes.
