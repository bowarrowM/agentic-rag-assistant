from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent_core import answer

app = FastAPI(title="RAG Assistant")

class Turn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[Turn] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    history: list[Turn]
@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = [turn.model_dump() for turn in request.history]
    text, sources = answer(request.question, history=history)

    new_history = (
        history
        + [{"role": "user", "content": request.question}]
        + [{"role": "assistant", "content": text}]
    )
    
    return ChatResponse(answer=text, sources=sources, history=new_history)