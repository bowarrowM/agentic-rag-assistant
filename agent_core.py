import chromadb
import ollama

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("nimbus_docs")

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer ONLY using information returned by the search_documents tool. "
    "If the answer is not in those results, say you don't know. "
    "Do not invent steps, numbers, or details that are not in the results."
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
    }
]


def search_documents(query):
    results = collection.query(query_texts=[query], n_results=3)
    chunks = results["documents"][0]
    sources = results["metadatas"][0]          # which file each chunk came from
    return chunks, sources


def answer(question, history=None):
    """Run the agent loop for one question. Returns (answer_text, sources_used)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history                     # earlier turns, for memory (Step 3)
    messages.append({"role": "user", "content": question})

    used_sources = []

    for step in range(5):
        response = ollama.chat(model=MODEL, messages=messages, tools=tools)
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message["content"], used_sources


        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if name == "search_documents":
                chunks, sources = search_documents(args["query"])
                for s in sources:
                    if s["source"] not in used_sources:
                        used_sources.append(s["source"])
                messages.append({"role": "tool", "content": "\n\n".join(chunks)})

    return "Sorry, I couldn't work that out.", used_sources

# if __name__ == "__main__":
#     text, sources = answer("how do I get my money back?")
#     print(text)
#     print("Sources:", sources)
