from fastapi import APIRouter

router = APIRouter(tags=["Health"])

@router.get("/healthcheck")
async def health_check():
    
    return {"status":"ok"}