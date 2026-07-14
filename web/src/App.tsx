import { useState, useRef, useEffect } from 'react'
import type { FormEvent } from 'react'
import './App.css'

// ---- Types that mirror the FastAPI backend contract ----
// Keeping these in sync with the Pydantic models in app.py is what makes
// the client "typed": the compiler catches a mismatch before runtime.
type Turn = { role: string; content: string }

type TraceStep = {
  type: string
  standalone?: string
  tool?: string
  args?: Record<string, unknown>
  sources?: string[]
  result?: Record<string, unknown>
  verdict?: string
}

type ChatResponse = {
  answer: string
  sources: string[]
  trace: TraceStep[]
  history: Turn[]
}

// A message as we render it in the UI (carries its own sources + trace)
type Message = {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  trace?: TraceStep[]
}

const API = 'http://127.0.0.1:8000'

// The single typed API client. Everything the UI knows about the backend
// goes through here.
async function chat(question: string, history: Turn[]): Promise<ChatResponse> {
  const res = await fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history }),
  })
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}

// Collapsible panel that renders the agent's reasoning steps.
function TracePanel({ trace }: { trace: TraceStep[] }) {
  const [open, setOpen] = useState(false)
  if (!trace || trace.length === 0) return null

  return (
    <div className="trace">
      <button className="trace-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? '▾' : '▸'} Agent trace ({trace.length} steps)
      </button>

      {open && (
        <div className="trace-body">
          {trace.map((step, i) => (
            <div className="trace-step" key={i}>
              {step.type === 'condense' && (
                <>
                  <span className="trace-tag">condense</span>
                  <span className="trace-text">{step.standalone}</span>
                </>
              )}

              {step.type === 'tool_call' && (
                <>
                  <span className="trace-tag tool">{step.tool}</span>
                  <code className="trace-text">{JSON.stringify(step.args)}</code>
                  {step.sources && step.sources.length > 0 && (
                    <div className="trace-sub">→ sources: {step.sources.join(', ')}</div>
                  )}
                  {step.result && (
                    <div className="trace-sub">→ result: {JSON.stringify(step.result)}</div>
                  )}
                </>
              )}

              {step.type === 'verify' && (
                <>
                  <span
                    className={`trace-tag verify ${
                      step.verdict === 'SUPPORTED' ? 'ok' : step.verdict === 'UNSUPPORTED' ? 'bad' : ''
                    }`}
                  >
                    verify
                  </span>
                  <span className="trace-text">
                    {step.verdict === 'SUPPORTED'
                      ? '✓ answer grounded in sources'
                      : step.verdict === 'UNSUPPORTED'
                        ? '✗ not fully supported by sources'
                        : '~ inconclusive'}
                  </span>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [history, setHistory] = useState<Turn[]>([]) // sent back to the server each turn
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to the newest message.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const question = input.trim()
    if (!question || loading) return

    setMessages((m) => [...m, { role: 'user', content: question }])
    setInput('')
    setLoading(true)

    try {
      const data = await chat(question, history)
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: data.answer, sources: data.sources, trace: data.trace },
      ])
      setHistory(data.history) // server returns the full transcript; we hold it
    } catch {
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: 'Error reaching the assistant. Is the API running on :8000?' },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          Kitty Pay <span>· Support Assistant</span>
        </div>
        <div className="sub">
          Grounded answers over the help center · escalates disputes to a human
        </div>
      </header>

      <div className="chat">
        {messages.length === 0 && (
          <div className="empty">
            Ask about fees, transfers, or refunds — or report an unauthorized payment.
          </div>
        )}

        {messages.map((m, i) => (
          <div className={`row ${m.role}`} key={i}>
            <div className="bubble">{m.content}</div>

            {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
              <div className="chips">
                {m.sources.map((s, j) => (
                  <span className="chip" key={j}>
                    {s}
                  </span>
                ))}
              </div>
            )}

            {m.role === 'assistant' && m.trace && <TracePanel trace={m.trace} />}
          </div>
        ))}

        {loading && (
          <div className="row assistant">
            <div className="bubble thinking">Thinking…</div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question…"
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}

export default App
