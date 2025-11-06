from fastapi import APIRouter, Depends, HTTPException, Request
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from app.services.llm_client import LLMClient, UpstreamTimeout, UpstreamHTTPError
from app.core.settings import Settings

router = APIRouter(prefix="", tags=["Chat"])

def get_llm_client() -> LLMClient:
    settings = Settings()
    return LLMClient(settings)

@router.post("/v1/chat/completions", response_model = ChatCompletionResponse)
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    llm_client: LLMClient = Depends(get_llm_client)):

    if req.stream:
        raise HTTPException(status_code = 400, detail={"type": "bad_request", "message": "stream=true not supported yet"})

    try:
        res = await llm_client.chat_completion(req)
        return res
    except UpstreamTimeout as e:
        raise HTTPException(
            status_code=504,
            detail={"type": "timeout", "message": str(e), "request_id": getattr(request.state, "request_id", "-")},
        )
    except UpstreamHTTPError as e:
        if 400 <= e.status_code < 500:
            raise HTTPException(
                status_code=400,
                detail={"type": "upstream_4xx", "message": e.body, "request_id": getattr(request.state, "request_id", "-")},
            )
        raise HTTPException(
            status_code=502,
            detail={"type": "upstream_5xx", "message": e.body, "request_id": getattr(request.state, "request_id", "-")},
        )
    finally:
        # Ensure httpx client closed (prevents connection leaks)
        await llm_client.aclose()

