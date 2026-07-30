import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.db import init_db
from app.services.gpu_service import gpu_monitor_loop

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    task = asyncio.create_task(gpu_monitor_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.routes import (  # noqa: E402
    checkpoints,
    datasets,
    gpu,
    inference,
    models,
    system,
    tokenizer,
    training,
)

app.include_router(system.router, prefix=settings.api_prefix + "/system", tags=["system"])
app.include_router(models.router, prefix=settings.api_prefix + "/models", tags=["models"])
app.include_router(datasets.router, prefix=settings.api_prefix + "/datasets", tags=["datasets"])
app.include_router(training.router, prefix=settings.api_prefix + "/training", tags=["training"])
app.include_router(checkpoints.router, prefix=settings.api_prefix + "/checkpoints", tags=["checkpoints"])
app.include_router(inference.router, prefix=settings.api_prefix + "/inference", tags=["inference"])
app.include_router(gpu.router, prefix=settings.api_prefix + "/gpu", tags=["gpu"])
app.include_router(tokenizer.router, prefix=settings.api_prefix + "/tokenizer", tags=["tokenizer"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": settings.app_name}
