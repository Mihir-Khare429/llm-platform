from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import time
import uuid

class ChatMessage(BaseModel) :
    role : Literal["system","user","assistant"] = "user"
    content : str

class ChatCompletionRequest(BaseModel) : 
    model : str
    messages : List[ChatMessage]
    temperature : Optional[float] = 0.2
    top_p : Optional[float] = 1.0
    max_token : Optional[int] = 256
    seed : Optional[int] = None
    user : Optional[str] = None
    stream : Optional[bool] = False

class ChoiceMessage(BaseModel) : 
    role : Literal["assistant"] = "assistant"
    content : str

class ChatChoice(BaseModel) : 
    index : int
    message : ChoiceMessage
    finish_reason : Optional[str] = None

class Usage(BaseModel) : 
    prompt_token : int = 0
    completion_tokens : int = 0
    total_tokens : int = 0

class ChatCompletionResponse(BaseModel) : 
    id : str = Field(default_factory = lambda : f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object : str = "chat.completion"
    created : int = Field(default_factory = lambda : int(time.time()))
    model : str
    choices : List[ChatChoice]
    usage : Usage