FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    dumb-init \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#include both app and src directories in path
ENV PYTHONPATH=/app:/app/src

RUN mkdir -p model_artifacts logs

EXPOSE 8192

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8192/health || exit 1
# Start application
ENTRYPOINT ["dumb-init", "--"]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8192"]
