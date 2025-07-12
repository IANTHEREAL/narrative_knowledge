"""
Main FastAPI application for Narrative Knowledge API.
"""

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.knowledge import router as knowledge_router
from api.memory import router as memory_router
from api.ingest import router as ingest_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Narrative Knowledge API",
    description="API for narrative knowledge extraction and graph building",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(knowledge_router)
app.include_router(memory_router)
app.include_router(ingest_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Custom HTTP exception handler to ensure consistent error responses.

    Args:
        request: The incoming request
        exc: The HTTP exception raised

    Returns:
        JSON response with standardized error format
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": (
                    exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                ),
                "status_code": exc.status_code,
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    General exception handler for unhandled exceptions.

    Args:
        request: The incoming request
        exc: The exception raised

    Returns:
        JSON response with error information
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
            }
        },
    )


@app.get("/")
async def root():
    """
    Root endpoint providing basic API information.

    Returns:
        Basic API information and health status
    """
    return {
        "message": "Narrative Knowledge API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {"docs": "/docs", "redoc": "/redoc"},
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service health status
    """
    return {"status": "healthy", "service": "narrative-knowledge-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
