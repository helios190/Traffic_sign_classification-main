# 1️⃣ Builder stage (pure-Python only)
FROM python:3.10-slim AS builder
WORKDIR /wheels

# Copy only the "pure" requirements
# We’ll build wheels for everything *except* tensorflow-cpu, tflite-runtime, etc.
COPY requirements.txt .
RUN grep -vE '^(tensorflow-cpu|tensorflow-macos|tensorflow-metal|tflite-runtime)' requirements.txt \
    > requirements-pure.txt && \
    pip wheel --wheel-dir /wheels -r requirements-pure.txt

# 2️⃣ Runtime stage
FROM python:3.10-slim
WORKDIR /app

# Copy in the locally-built pure-Python wheels
COPY --from=builder /wheels /wheels
COPY requirements.txt .

# First, install the platform‐specific packages that we cannot wheel here:
RUN pip install --no-cache-dir \
      tensorflow-cpu==2.13.1 \
      tflite-runtime==2.13.0

# Then install everything else from our wheelhouse:
RUN pip install --no-index --find-links=/wheels -r requirements.txt

# Copy the rest of your app
COPY . .

ENV PYTHONUNBUFFERED=1 \
    ARTEFACT_PATH=models/v2025-06-27/model_int8.tflite

EXPOSE 8080
CMD ["uvicorn", "src.wrapper:app", "--host", "0.0.0.0", "--port", "8080"]
