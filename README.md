# AI Image Detector API

A FastAPI backend that will serve a trained model for detecting **AI-generated images** through a REST API.

> **Status:** Initial setup. Only a health-check endpoint (`GET /`) is available. The `/predict` endpoint is not implemented yet.

## Architecture

```
ai-image-detector-api/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI application & routes
├── models/
│   └── calibrated_svm.pkl   # Trained calibrated RBF SVM (exported from Kaggle)
├── .gitignore
├── requirements.txt
└── README.md
```

- **API layer:** FastAPI served by Uvicorn.
- **Model artifact:** `models/calibrated_svm.pkl` is a calibrated RBF SVM classifier trained in Kaggle. It is loaded with `joblib` / `scikit-learn`. Do not modify or replace it.
- **Feature extraction (planned):** `torch` + `transformers` will generate image embeddings that the SVM then classifies; `pillow` handles image decoding and `python-multipart` handles file uploads.

## Setup

### 1. Create a virtual environment

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

### 2. Install requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add the model artifact

Copy your trained `calibrated_svm.pkl` into the `models/` directory.

## Running the server

```bash
uvicorn app.main:app --reload
```

The server runs at `http://127.0.0.1:8000`. Interactive docs are at `http://127.0.0.1:8000/docs`.

## Testing `GET /`

**curl:**
```bash
curl http://127.0.0.1:8000/
```

**PowerShell:**
```powershell
Invoke-RestMethod http://127.0.0.1:8000/
```

Expected response:
```json
{"status": "ok", "message": "AI Image Detector API is running"}
```

## Security

Never commit secrets. Keep credentials in a local `.env` file. `.env` and `.env.*` are already in `.gitignore`.
