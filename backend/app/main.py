"""FastAPI application entry point for the Phase 0 scaffold."""

from fastapi import FastAPI

from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.0.1")


@app.get("/health", tags=["system"])
def healthcheck() -> dict[str, str]:
    """Provide a lightweight container health endpoint."""

    return {"status": "ok", "environment": settings.app_env}

