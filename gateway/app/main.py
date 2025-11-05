from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health
from app.core.logging import configure_logging
from app.core.settings import Settings

def get_app() -> FastAPI:
    settings = Settings()
    configure_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title = "LLM_Platform",
        version = "0.1.0",
        description = "Unified LLM inference gateway"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    app.include_router(health.router)

    @app.get("/")
    async def root():

        return {
            "message": "LLM Gateway running", 
            "env": settings.APP_ENV
        }

    return app