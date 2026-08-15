from pathlib import Path
import re
import chromadb

# LOAD

files = Path("documents").glob("*.md")

documents = []
for file in files:
    text = file.read_text()
    documents.append((file.name, text))
    print(f"Loaded {file.name} - {len(text)} characters")


# CHUNK
# Split on blank lines so a Q&A pair / paragraph / list stays intact, and carry
# the current section heading into each chunk so it's self-contained.

def chunk_markdown(text):
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks = []
    heading = ""
    for block in blocks:
        if block.lstrip().startswith("#"):
            heading = block.lstrip("# ").strip()
            continue
        chunks.append(f"{heading}\n{block}" if heading else block)
    return chunks


all_chunks = []
for name, text in documents:
    for chunk in chunk_markdown(text):
        all_chunks.append((name, chunk))
print(f"Total chunks: {len(all_chunks)}")


# STORE

client = chromadb.PersistentClient(path="chroma_db")
# Rebuild cleanly so re-chunking never leaves stale chunks behind.
try:
    client.delete_collection("kitty_docs")
except Exception:
    pass
collection = client.get_or_create_collection("kitty_docs")

texts = [chunk for (name, chunk) in all_chunks]
ids = [f"chunk-{i}" for i in range(len(all_chunks))]
metadatas = [{"source": name} for (name, chunk) in all_chunks]

collection.upsert(documents=texts, ids=ids, metadatas=metadatas)

print(f"Stored {collection.count()} chunks in the vector database")
