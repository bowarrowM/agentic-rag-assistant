from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from agent_core import answer

app = FastAPI(title="Kitty Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[Turn] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    trace: list[dict]=[]
    history: list[Turn]
@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = [turn.model_dump() for turn in request.history]
    text, sources, trace = answer(request.question, history=history)

    new_history = (
        history
        + [{"role": "user", "content": request.question}]
        + [{"role": "assistant", "content": text}]
    )
    
    return ChatResponse(answer=text, sources=sources, trace=trace, history=new_history)