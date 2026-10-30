"""Vercel entrypoint for the lightweight FastAPI deployment."""

from app.main import app

__all__ = ["app"]
