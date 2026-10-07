from fastapi import FastAPI
from app.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.0.1"
)


@app.get('/health')
def health_check():
    return {"status": "healthy",
            "service": settings.app_name}
