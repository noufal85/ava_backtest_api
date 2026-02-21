"""Entry point — run with: python -m src.main"""
import uvicorn

from src.api.v2.app import app  # noqa: F401

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8201, reload=True)
