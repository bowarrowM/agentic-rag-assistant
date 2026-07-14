import chromadb
import ollama
import uuid
import json

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("kitty_docs")

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are the Kitty Pay support assistant. Kitty Pay is a digital wallet for "
    "sending money, topping up from a card, and paying merchants. "
    "Answer ONLY using information returned by the search_documents tool. "
    "If the answer is not in those results, say you don't know. "
    "Do not invent fees, limits, numbers, or policies that are not in the results. "
    "If the user reports an unauthorized payment, a transfer to the wrong person, a "
    "dispute, or asks for a human — open a support ticket with create_support_ticket. "
    "Never mention tool or function names to the user; just help them naturally. "
    "Always call search_documents before answering any question about fees, limits, "
    "refunds, transfers, security, or policies — never answer these from memory. "

)

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the knowledge base for information about the product, billing, refunds, security, or support.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function":{
            "name":"create_support_ticket",
            "description":" Open a support ticket when the user reports an unauthorized payment, a dispute, a transfer sent to the wrong person, or asks for a human - or when the knowledge base has no answer. Do NOT use it for questions the knowledge base can answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type":"string", "description": "One-line summary of the user's issue"},
                    "email": {"type": "string", "description":"User's email if they gave one, else 'unknown'"},
                },
                "required": ["summary"],
            },
        }, 
    },
]


def search_documents(query):
    results = collection.query(query_texts=[query], n_results=3)
    chunks = results["documents"][0]
    sources = results["metadatas"][0]          # which file each chunk came from
    return chunks, sources

def create_support_ticket(summary, email="unknown"):
    ticket_id = "TICK-" + uuid.uuid4().hex[:6].upper()
    return {"ticket_id": ticket_id, "summary": summary, "email":email, "status": "open"}
    
def verify_answer(question, answer_text, context):
    """LLM-as-judge: is the answer grounded in the retrieved context?"""
    prompt = (
        "You check whether an ANSWER is grounded in the CONTEXT.\n"
        "Reply SUPPORTED if the answer's factual claims appear in or directly follow from "
        "the CONTEXT. Reply UNSUPPORTED only if the answer states a fact that is absent "
        "from or contradicts the CONTEXT.\n"
        "Reply with exactly one word: SUPPORTED or UNSUPPORTED.\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer_text}\n\nVerdict:"
    )
    resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
    verdict = resp["message"]["content"].strip().upper()
    if "UNSUPPORTED" in verdict:
        return "UNSUPPORTED"
    if "SUPPORTED" in verdict:
        return "SUPPORTED"
    return "UNSURE"



def condense_question(question, history):
      """Rewrite a follow-up into a standalone question using the conversation."""
      if not history:
          return question                      # first turn — nothing to condense

      convo = "\n".join(f"{t['role']}: {t['content']}" for t in history)
      prompt = (
          "Given the conversation below, rewrite the user's follow-up question "
          "into a standalone question that makes sense on its own. "
          "Reply with ONLY the rewritten question and nothing else.\n\n"
          f"Conversation:\n{convo}\n\n"
          f"Follow-up: {question}\n\n"
          "Standalone question:"
      )
      resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
      return resp["message"]["content"].strip()



def answer(question, history=None):
    """Run the agent loop for one question. Returns (answer_text, sources_used)."""
    standalone = condense_question(question, history)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history
    messages.append({"role": "user", "content": standalone})

    used_sources = []
    trace = []
    retrieved_context = []
    trace.append({"type":"condense", "standalone":standalone})

    for step in range(5):
        response = ollama.chat(model=MODEL, messages=messages, tools=tools, options={"temperature": 0})
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            final = message["content"]
            if retrieved_context:
                verdict = verify_answer(standalone, final, "\n\n".join(retrieved_context))
                trace.append({"type": "verify", "verdict": verdict})
            return final, used_sources, trace


            

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]

            if name == "search_documents":
                chunks, sources = search_documents(args["query"])
                for s in sources:
                    if s["source"] not in used_sources:
                        used_sources.append(s["source"])
                messages.append({"role": "tool", "content": "\n\n".join(chunks)})
                retrieved_context.extend(chunks)
                trace.append({"type":"tool_call", "tool":"search_documents", "args":args, "sources":[s["source"] for s in sources]})

            elif name == "create_support_ticket":
                ticket = create_support_ticket(
                    summary=args.get("summary"),
                    email=args.get("email", "unknown"),
                )
                messages.append({"role": "tool", "content": json.dumps(ticket)})
                trace.append({"type":"tool_call", "tool":"create_support_ticket", "args":args, "result": ticket,})


    return "Sorry, I couldn't work that out.", used_sources, trace


# if __name__ == "__main__":
#     text, sources = answer("how do I get my money back?")
#     print(text)
#     print("Sources:", sources)
