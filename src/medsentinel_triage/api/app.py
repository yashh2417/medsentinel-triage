"""FastAPI application configuring CORS, exception handlers, API routers, and frontend static files."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from medsentinel_triage.api.routes_inference import router as inference_router
from medsentinel_triage.api.routes_preprocess import router as preprocess_router
from medsentinel_triage.base import InferenceError, InputValidationError, ModelLoadError
from medsentinel_triage.config import configure_logging

configure_logging()

FRONTEND_DIR: Path = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="MedSentinel Clinical Triage AI API",
    version="1.0.0",
    description=(
        "Modular FastAPI backend for multimodal clinical file preprocessing (`input_processor.py`), "
        "individual model inference, and stateful `LangGraph` HITL triage orchestration."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(InputValidationError)
async def input_validation_exception_handler(_request: Request, exc: InputValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"status": "error", "detail": str(exc)})


@app.exception_handler(ModelLoadError)
async def model_load_exception_handler(_request: Request, exc: ModelLoadError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})


@app.exception_handler(InferenceError)
async def inference_exception_handler(_request: Request, exc: InferenceError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc)})


app.include_router(preprocess_router)
app.include_router(inference_router)


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css", include_in_schema=False)
def serve_css() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def serve_js() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
