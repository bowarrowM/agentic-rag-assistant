import chromadb
import ollama

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("nimbus_docs")

question = "whats the weather like?"

# retrieve
results = collection.query(
    query_texts=[question],   # Chroma embeds this with the same local model
    n_results=3,              # give the 3 closest chunks (top-k, k=3)
)
retrieved_chunks = results["documents"][0]


# augment

context = "\n\n".join(retrieved_chunks)
prompt = f"""Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

# generate

response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])

print(f"Question: {question}\n")
print("Answer:")
print(response["message"]["content"])