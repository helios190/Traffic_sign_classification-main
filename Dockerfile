FROM python:3.10-slim AS builder
WORKDIR /tmp
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

##########################
# 2️⃣  Runtime image      #
##########################
FROM python:3.10-slim
WORKDIR /app

# copy wheels & requirements first
COPY --from=builder /wheels /wheels
COPY requirements.txt .

# install from local wheelhouse
RUN pip install --no-index --find-links=/wheels -r requirements.txt

# now bring in the rest of the source code
COPY . .

ENV PYTHONUNBUFFERED=1 \
    ARTEFACT_PATH=Traffic.h5

EXPOSE 8080
CMD ["uvicorn", "src.wrapper:app", "--host", "0.0.0.0", "--port", "8080"]