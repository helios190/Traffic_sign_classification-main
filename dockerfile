FROM python:3.10-slim AS builder
WORKDIR /tmp
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:3.10-slim
WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels \
        tflite-runtime==2.13.0 \
        fastapi uvicorn python-multipart \
        numpy opencv-python-headless pydantic

COPY . .
ENV PYTHONBUFFERED=1 \
    ARTEFACT_PATH=./models/v2025-06-27/model_int8.tflite

EXPOSE 8080
CMD ["uvicorn", "src.wrapper:app", "--host", "0.0.0.0", "--port", "8080"]