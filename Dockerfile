FROM python:3.11.11-slim-bookworm

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir .

USER appuser

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
