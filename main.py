from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
import sys
from pathlib import Path
from routers import finder, pharmacy_locator

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

# Include routers - removing the /api prefix for simplicity
app.include_router(finder.router, prefix="/finder", tags=["finder"])
app.include_router(pharmacy_locator.router, prefix="/locator", tags=["pharmacy-locator"])

@app.get("/")
async def root():
    return {"message": "Welcome to GenericBro API"}

@app.get("/routes")
def get_routes():
    """List all available routes for debugging."""
    routes = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            routes.append({
                "path": route.path,
                "name": route.name,
                "methods": route.methods
            })
    return routes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
