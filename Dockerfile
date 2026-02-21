FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini .

EXPOSE 8201
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8201"]
