# Web analyzer. Runs as-is on Hugging Face Spaces (Docker SDK), Render or Fly.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 app
WORKDIR /app
COPY --chown=app . .
RUN pip install --no-cache-dir ".[detect]"
# Fetch the OCR models at build time (the app runs as a user that cannot write
# site-packages). Reading a real line of text loads both detector and recognizer.
RUN python -m observatory.detect.frametext --warmup

USER app
ENV REPORTS_DIR=/tmp/reports \
    EXAMPLES_DIR=/app/examples/reports \
    BENCHMARKS_FILE=/app/docs/benchmarks.json \
    DATASET_FILE=/app/eval/dataset.csv \
    ALLOW_UPLOAD=0 \
    TRUST_PROXY=1 \
    DOCS_ROOT=/app \
    PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn observatory.detect.web:app --host 0.0.0.0 --port ${PORT}"]
