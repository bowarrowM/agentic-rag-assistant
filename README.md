# RAG Assistant

A conversational, retrieval-augmented assistant that answers questions over a document set
using a **local LLM** (via Ollama) and a **local vector store** (Chroma) — no paid APIs,
no API keys, runs entirely on your machine.

Every answer is **grounded** in retrieved source documents and reports **which files it used**,
so retrieval is inspectable. The assistant holds a **multi-turn conversation** and rewrites
follow-up questions into standalone queries so retrieval stays accurate across a dialogue.

Built to be **hardware-ready**: the retrieval core is exposed over an HTTP API, so a physical
sensor/component can plug in later as another client.

## Demo

A small chat UI is served at `/`, and an interactive API (Swagger) at `/docs`.

```
You:  How do I get a refund?
Bot:  You can request a full refund within 14 days of any payment, no questions asked —
      contact support@nimbus.example. Refunds are processed within 5–10 business days.
      Sources: product_faq.md, support_policies.md

You:  How long does it take?
Bot:  It typically takes 5 to 10 business days to receive a refund after requesting one.
      Sources: support_policies.md, product_faq.md
```

The second question never says "refund" — the assistant condenses it into a standalone
question using the conversation history, so retrieval still finds the right chunk.

## Architecture

```
documents/ ──▶ ingest ──▶ chunk (overlap) ──▶ embed (local) ──▶ Chroma vector store
                                                                      │
follow-up + history ──▶ condense to standalone question              │
                                    │                                │
                                    ▼                                │
                        embed ──▶ retrieve top-k chunks ◀────────────┘
                                    │
                                    ▼
              local LLM (Ollama) + tool-calling ──▶ grounded answer + sources
                                    │
                                    ▼
                  FastAPI:  GET /  (chat UI)  ·  POST /chat  (JSON API)
```

## Key features

- **Agentic tool-calling** — the LLM decides when to search the knowledge base, via an
  Ollama tool schema, inside a loop with a **step cap** guardrail against runaway calls.
- **Grounding guardrail** — a system prompt constrains the model to answer only from
  retrieved context and to say "I don't know" otherwise, reducing hallucination.
- **Multi-turn memory** — conversation history is passed with each request (stateless
  server; the client holds the transcript).
- **Query rewriting (condense-question)** — follow-ups like "how long does it take?" are
  rewritten into standalone questions before retrieval, so context-dependent turns work.
- **Source attribution** — every answer reports which document(s) it drew from.
- **Fully local & free** — Ollama for generation/tool-calling, Chroma for vectors +
  embeddings. No external API, no key, no cost.

## Stack

- **Ollama** (`llama3.2:3b`) — local LLM, generation + tool-calling
- **Chroma** — local vector database with built-in local embeddings
- **FastAPI + Uvicorn** — HTTP API and static chat UI
- **Pydantic** — typed request/response contracts

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install Ollama (https://ollama.com), then pull the model:
ollama pull llama3.2:3b
```

## Usage

```bash
# 1. Ingest the documents into the vector store (run once, or after editing documents/)
python ingest.py

# 2. Start the server
uvicorn app:app --reload

# 3. Open the chat UI
#    http://127.0.0.1:8000/          → web chat
#    http://127.0.0.1:8000/docs      → interactive API
```

Swap the files in `documents/` for your own knowledge base, re-run `python ingest.py`,
and the assistant answers over your content.

## Project layout

```
ingest.py        Load → chunk (with overlap) → embed → store in Chroma
agent_core.py    The agent: condense-question, tool-calling loop, grounding, sources
app.py           FastAPI app: GET / (chat UI) and POST /chat (JSON API)
index.html       Minimal web chat client
documents/       The knowledge base (Markdown)
```

## Notes & limitations

- Uses a small 3B local model for zero cost; it occasionally drifts beyond the source
  text despite the grounding prompt. A larger model (or an added answer-verification /
  LLM-as-judge pass) would tighten this — a deliberate cost/quality trade-off.
- Chunking is fixed-size with overlap. Semantic or structure-aware chunking would improve
  retrieval on longer, mixed-topic documents.

## Roadmap

- [x] Ingestion (load → chunk → embed → store)
- [x] Retrieval + grounded Q&A with sources
- [x] Agentic tool-calling loop (step-cap + grounding guardrails)
- [x] Multi-turn memory + query rewriting
- [x] FastAPI `/chat` API + web chat UI
- [ ] Hardware client (physical component talks to `/chat`)
