import chromadb
import ollama

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("nimbus_docs")

def search_documents(query):
    results = collection.query(query_texts=[query], n_results=3)
    return "\n\n".join(results["documents"][0])

# tool description to the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the Nimbus knowledge base for information about the product, billing, refunds, security, or support.",
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

question = "how do I get my money back?"

# agent loop
#messages = [{"role": "user", "content": question}]

messages = [
    {
        "role": "system",
        "content": (
            "You are a support assistant for Nimbus. "
            "Answer ONLY using information returned by the search_documents tool. "
            "If the answer is not in those results, say you don't know. "
            "Do not invent steps, numbers, or details that are not in the results."
        ),  
    },
    {"role": "user", "content": question},
] 

for step in range(5):                    # step cap = guardrail against infinite loops
    response = ollama.chat(model="llama3.2:3b", messages=messages, tools=tools)
    message = response["message"]
    messages.append(message)             # record what the model just said

    tool_calls = message.get("tool_calls")
    if not tool_calls:                   # no tool call , gave its final answer
        print("\nFinal answer:")
        print(message["content"])
        break

    for call in tool_calls:              # or, run the tool it asked for
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        print(f"[Agent decided to call {name}({args})]")
        if name == "search_documents":
            result = search_documents(args["query"])
            messages.append({"role": "tool", "content": result})   # result feedback
else:
    print("\nStopped: hit the step cap (guardrail).")
