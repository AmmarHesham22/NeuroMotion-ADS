import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ai_service.routers import inference

app = FastAPI(
    title="NeuroMotion-ADS AI Service",
    description="Python inference backend for the NeuroMotion-ADS foundation model.",
    version="1.0.0"
)

# Configure CORS for .NET backend and Flutter clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint to verify service status.
    """
    return {"status": "ok", "service": "NeuroMotion-ADS AI Service"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
