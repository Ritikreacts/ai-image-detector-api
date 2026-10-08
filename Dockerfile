FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Render sets $PORT; default to 10000 for local Docker builds
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
