FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy everything first so hatchling can resolve the src/ package layout
COPY pyproject.toml .
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini .

# Install dependencies (non-editable for production)
RUN pip install --no-cache-dir .

EXPOSE 8201
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8201"]
