"""
SentinelFetal FastAPI Main Entry Point
======================================
V3 Full-Stack Migration
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.routers import patients, simulation, websocket
from api.services.orchestrator_adapter import get_orchestrator_adapter
from api.services.broadcaster import get_broadcaster
from api.services.orchestrator_bridge import get_orchestrator_bridge
from api.models.schemas import HealthCheck

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("🚀 SentinelFetal API starting...")
    
    # Initialize orchestrator adapter (but don't start simulation)
    orchestrator = get_orchestrator_adapter()
    orchestrator.initialize(patient_count=settings.default_patient_count)
    logger.info("✅ Orchestrator adapter initialized")
    
    # Initialize and start WebSocket broadcaster
    broadcaster = get_broadcaster()
    await broadcaster.start()
    logger.info("✅ WebSocket broadcaster started")
    
    # Start orchestrator bridge for thread-safe data transfer
    bridge = get_orchestrator_bridge()
    await bridge.start_transfer(broadcaster)
    logger.info("✅ Orchestrator bridge transfer started")
    
    yield
    
    # Shutdown
    logger.info("👋 SentinelFetal API shutting down...")
    
    # Stop bridge first
    await bridge.stop_transfer()
    logger.info("✅ Orchestrator bridge stopped")
    
    # Stop broadcaster
    await broadcaster.stop()
    logger.info("✅ WebSocket broadcaster stopped")
    
    # Stop orchestrator
    orchestrator.shutdown()
    logger.info("✅ Orchestrator adapter shut down cleanly")


app = FastAPI(
    title="SentinelFetal API",
    description="Real-time CTG monitoring with AI classification",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(patients.router, prefix="/api/patients", tags=["patients"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["simulation"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


@app.get("/api/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    orchestrator = get_orchestrator_adapter()
    broadcaster = get_broadcaster()
    
    # Check if orchestrator and broadcaster are functional
    try:
        _ = orchestrator.get_status()
        stats = broadcaster.get_stats()
        if stats["running"]:
            status = "healthy"
        else:
            status = "degraded"
    except Exception:
        status = "degraded"
    
    return HealthCheck(
        status=status,
        version="3.0.0",
        timestamp=time.time(),
    )


@app.get("/")
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "SentinelFetal API v3.0.0",
        "docs": "/api/docs",
        "health": "/api/health",
        "websocket": "/ws/stream",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
