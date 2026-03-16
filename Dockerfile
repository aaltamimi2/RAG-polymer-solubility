FROM node:20-bookworm AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DATA_DIR=/app/data \
    ML_MODEL_DIR=/app/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY app_server.py export_manager.py output_models.py ./
COPY data ./data
COPY models ./models
COPY --from=frontend-build /app/frontend/build ./frontend/build

RUN pip install --no-cache-dir .

RUN mkdir -p /app/plots /app/reports

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
