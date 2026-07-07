from fastapi import FastAPI
from pydantic import BaseModel

from agent_core import answer

app = FastAPI(totle="RAG Assistant")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    text, sources = answer(request.question)
    return ChatResponse(answer=text, sources=sources)