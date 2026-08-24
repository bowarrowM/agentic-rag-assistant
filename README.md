# Kitty Pay — Support Assistant

An **agentic RAG** customer-support assistant for **Kitty Pay**, a fictional digital-wallet
fintech. It answers account, fee, transfer, refund, and security questions **grounded strictly
in the help-center docs**, and **escalates disputes and unauthorized-payment reports to a human**
by opening a ticket.

Runs entirely **locally and free** — a local LLM (Ollama) and a local vector store (Chroma), no
paid APIs, no keys.

> **Why a fintech, and why grounding matters.** In a payments product, a support bot that invents
> a fee, a limit, or a refund policy is a compliance and trust failure — not a cosmetic bug. So
> grounding, source attribution, self-verification, and human escalation aren't nice-to-haves
> here; they're the requirements the whole design is built around.

## What it does

- **Answers from the knowledge base** — grounded in `documents/`, and reports which files it used.
- **Routes between two tools** — it *decides* whether to search the docs or open a support ticket.
  A question about fees → search; "someone charged my account without permission" → escalate.
- **Verifies its own answers** — an LLM-as-judge checks each grounded answer against the retrieved
  context and surfaces a `SUPPORTED` / `UNSUPPORTED` verdict.
- **Shows its work** — the API returns an **agent trace** (condense → tool calls → verification),
  rendered live in the UI, so every decision is inspectable.

## Demo

```
You:  how much can I send per day, and how long do refunds take?
Bot:  Unverified accounts can send up to €500 per day; fully verified accounts up to €10,000 per
      day and €50,000 per month. Merchant refunds are returned to your balance in 5–10 business days.
      Sources: product_faq.md, support_policies.md
      Trace:  condense · search_documents · verify ✓ grounded

You:  someone made a payment from my account I didn't authorize
Bot:  I've opened a support ticket (TICK-410D12) for our team to investigate. Please freeze your
      wallet from Settings → Security in the meantime.
      Trace:  condense · create_support_ticket → {ticket_id: TICK-410D12, status: open}
```

The React client (`web/`) renders each answer with its **source chips** and an expandable
**agent-trace panel**.

## Architecture

```
documents/ ─▶ ingest ─▶ chunk (overlap) ─▶ embed (local) ─▶ Chroma vector store
                                                                   │
user + history ─▶ condense to standalone question                 │
                              │                                    │
                              ▼          agent loop (step-capped)  │
                    local LLM decides which tool: ─────────────────┤
                              │                                    │
              ┌───────────────┴────────────────┐                  │
              ▼                                 ▼                  │
      search_documents ◀── retrieve top-k ──────┘         create_support_ticket
              │                                                    │
              ▼                                                    ▼
      grounded answer ─▶ verify (LLM-as-judge, advisory)     ticket {id, status}
              │
              ▼
   FastAPI  POST /chat  ─▶  { answer, sources, trace, history }
              │
              ▼
   React + TypeScript SPA  (chat · source chips · agent-trace panel)
```

## Key features

- **Agentic tool-calling with real routing** — two tools (`search_documents`,
  `create_support_ticket`) so the model makes a genuine decision each turn, inside a loop with a
  **step-cap** guardrail against runaway calls.
- **Grounding guardrail** — a system prompt constrains the model to answer only from retrieved
  context and to say "I don't know" otherwise.
- **Answer verification (LLM-as-judge)** — a second model call checks the answer against the
  retrieved context; the verdict is surfaced in the trace (see *Design decisions* for why it's
  advisory, not blocking).
- **Human escalation** — disputes and unauthorized-payment reports open a ticket instead of being
  answered from docs, modeling the real fintech pattern of handing sensitive money issues to a person.
- **Query rewriting (condense-question)** — follow-ups are rewritten into standalone questions
  before retrieval, so multi-turn conversation works.
- **Agent trace** — the pipeline's steps are returned to the client and rendered, making the
  agent's reasoning observable rather than a black box.
- **Source attribution** — every grounded answer reports which document(s) it drew from.
- **Fully local & free** — Ollama for generation/tool-calling/judging, Chroma for vectors + embeddings.

## Design decisions

- **Verification is advisory, not blocking.** An early version hard-gated on the judge and fell
  back to "I can't confirm that" whenever it returned `UNSUPPORTED`. Testing showed a small local
  judge is **noisy** — it rejected correct, grounded answers (false negatives). Hard-gating on an
  unreliable judge makes the product *worse*, so the verdict is surfaced as an **advisory signal**
  in the trace. A stronger judge model would be needed to gate on it in production.
- **Stateless server, client-held transcript.** The API is stateless; the client passes the
  conversation history each turn. Simpler to scale, and the client owns its own state.
- **Local 3B model for zero cost.** A deliberate cost/quality trade-off. The model occasionally
  drifts or skips the search tool despite the prompt; the grounding and verification layers exist
  precisely to catch that.

## Evaluation

Retrieval and agent behavior are measured with a small, **reproducible eval harness** (`eval/`)
over a hand-built set of **14 labeled questions** — answerable, ticket-worthy, and out-of-scope —
each tagged with its expected behavior, gold source document, and the specific fact the answer
must contain. Improvements here are made *against the harness*, not by eyeballing outputs.

**Retrieval** — `python eval/run_eval.py` (fast, no LLM required):

| Metric (top-k = 3) | Result |
|---|---|
| Source hit-rate | 11/11 (100%) |
| Mean fact recall | 0.91 |
| Perfect fact recall | 10/11 (91%) |

Moving from a naive fixed-size split to overlapping chunking lifted mean fact recall@3 from an
earlier **0.73 to 0.91** — a measured change, not a guess.

**End-to-end** — `python eval/run_eval.py --full` (needs Ollama) additionally scores:

- **Tool-routing accuracy** — did the agent search vs. open a ticket vs. decline, as expected?
- **Faithfulness** — share of grounded answers the LLM-as-judge marks `SUPPORTED` against context.
- **Answer correctness** — does the final answer contain the required facts?

Every metric reproduces from the repo root with one command.

## Stack

- **Backend:** Python · FastAPI + Uvicorn · Pydantic (typed request/response contracts)
- **LLM / RAG:** Ollama (`llama3.2:3b`) · Chroma (local vectors + embeddings)
- **Frontend:** React + TypeScript (Vite) · typed API client · CSS
- **Infra:** Docker + Docker Compose (backend, frontend/nginx, Ollama) · one-command `setup.sh`

## Run with Docker (recommended)

Everything is containerized — the FastAPI backend, the React frontend (nginx), and the Ollama
LLM runtime. One command builds the images, pulls the model, ingests the docs, and starts it all:

```bash
./setup.sh
```

Then open **http://localhost:5173** for the chat UI (API docs at **http://localhost:8000/docs**).
First run downloads the base images and the ~2GB model, so give it a few minutes; later runs are fast.

```bash
docker compose down       # stop
docker compose down -v    # stop and wipe the model + vector store
```

## Run manually (without Docker)

**Backend** (terminal 1):
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install Ollama (https://ollama.com), then pull the model:
ollama pull llama3.2:3b

python ingest.py          # build the vector store (run once, or after editing documents/)
uvicorn app:app --reload  # API on http://127.0.0.1:8000  (Swagger at /docs)
```

**Frontend** (terminal 2):
```bash
cd web
npm install
npm run dev                # UI on http://localhost:5173
```

Swap the files in `documents/` for your own knowledge base, re-run `python ingest.py`, and the
assistant answers over your content.

## Project layout

```
ingest.py            Load → chunk (with overlap) → embed → store in Chroma
agent_core.py        The agent: condense-question, tool routing, verification, trace
app.py               FastAPI: POST /chat  → { answer, sources, trace, history }  (+ CORS)
documents/           The knowledge base (Markdown help-center docs)
web/                 React + TypeScript client (+ its own Dockerfile + nginx)
Dockerfile           Backend image (FastAPI + agent)
docker-compose.yml   Orchestrates ollama + backend + frontend
setup.sh             One command: build → pull model → ingest → run
```

## Limitations & next steps

- Small local judge is noisy — a stronger model would allow hard-gating on the verdict.
- Fixed-size chunking with overlap; semantic or structure-aware chunking + reranking would
  improve retrieval precision on longer, mixed-topic docs.
- Tickets are a mock (a real deployment would call a ticketing backend such as Zendesk/Jira).
