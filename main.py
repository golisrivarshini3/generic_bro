from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
from routers import finder

# Create FastAPI app
app = FastAPI(
    title="GenericBro API",
    description="Backend API for GenericBro application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(finder.router, prefix="/api/finder", tags=["finder"])

@app.get("/")
async def root():
    return {"message": "Welcome to GenericBro API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
