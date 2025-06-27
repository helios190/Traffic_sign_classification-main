FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel -r requirements.txt -w /wheels

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r /app/requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 ARTEFACT_PATH=models/Traffic.h5
EXPOSE 8000
CMD ["python", "clientApp.py"]        # ganti gunicorn kalau sudah ada
