# RAG Agent (local, tool-calling)

A retrieval-augmented chatbot that answers questions over a document set using a
**local LLM** (via Ollama) and a **local vector store** (Chroma) — no paid APIs,
runs entirely on your machine. Every answer reports which source chunks it used,
so retrieval is inspectable.

Built to be **hardware-ready**: the retrieval core is exposed over an API so a
physical sensor/component can plug in later.

## Architecture

```
documents/ ──▶ ingest ──▶ chunk ──▶ embed (local) ──▶ Chroma vector store
                                                            │
question ──▶ embed ──▶ retrieve top-k chunks ──────────────┘
                              │
                              ▼
                   local LLM (Ollama) + tool-calling ──▶ answer + sources
                              │
                              ▼
                       FastAPI /chat endpoint
```

## Stack
- **Chroma** — local vector database with built-in local embeddings (no API key)
- **Ollama** — runs the LLM locally (generation + tool-calling)
- **FastAPI** — serves the agent as an HTTP API

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install Ollama (https://ollama.com) and pull a model, e.g.:
ollama pull llama3.2:3b
```

## Usage
_(filled in as the project is built — Days 1–4)_

## Status
- [x] Day 1 — ingestion (load → chunk → embed → store)
- [x] Day 2 — retrieval + Q&A with sources
- [x] Day 3 — agentic tool-calling layer (with grounding guardrail)
- [ ] Day 4 — FastAPI endpoint + docs
