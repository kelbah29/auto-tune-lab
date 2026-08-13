from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.api.runs import router as runs_router  # noqa: E402
from app.core.hardware import available_ram_gb, has_cuda, has_mps, total_ram_gb  # noqa: E402

app = FastAPI(title="AutoTuneLab")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs_router)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "tinker_configured": bool(os.environ.get("TINKER_API_KEY")),
        "ram_available_gb": round(available_ram_gb(), 1),
        "ram_total_gb": round(total_ram_gb(), 1),
        "mps": has_mps(),
        "cuda": has_cuda(),
    }


# Single-service deploy: serve the built frontend (frontend/dist) directly from
# the API process rather than requiring a separate static host, so one Railway
# service covers the whole app. Mounted last and after all /api routes so it
# only catches non-API paths; a no-op locally where frontend/dist doesn't exist.
_frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
