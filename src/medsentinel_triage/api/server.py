"""CLI server runner for the MedSentinel Triage FastAPI Web Application (`uvicorn`)."""

from __future__ import annotations

import argparse
from typing import List, Optional

import uvicorn

from medsentinel_triage.api.routes_inference import EngineSingleton


def main(argv: Optional[List[str]] = None) -> None:
    """Starts the Uvicorn server hosting the MedSentinel Triage FastAPI Backend and Frontend UI."""
    parser = argparse.ArgumentParser(
        description="Start the MedSentinel Clinical Triage FastAPI Web Server."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port number to bind (default: 8000).")
    parser.add_argument(
        "--preload-models",
        action="store_true",
        help="Eagerly load all 6 pre-trained models into memory before accepting HTTP requests.",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    args = parser.parse_args(argv)

    if args.preload_models:
        EngineSingleton.get_engine()

    print(f"\nMedSentinel Triage Web Application : http://{args.host}:{args.port}")
    print(f"FastAPI Interactive Swagger Docs   : http://{args.host}:{args.port}/docs\n")
    uvicorn.run(
        "medsentinel_triage.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
