from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, query
from app.models.schemas import HealthResponse

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="RAG-based internal document Q&A system with RBAC and verification.",
    version="0.1.0",
)

# Allow the React frontend (added Week 3) to call this API during local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(query.router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")
