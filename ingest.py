from pathlib import Path 
import  chromadb

# LOAD

files = Path("documents").glob("*.md")

documents = [] 
for file in files:
    text = file.read_text( )
    documents.append((file.name, text))
    print(f"Loaded {file.name} - {len(text)} characters")


# CHUNK

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

all_chunks = []
for name, text in documents:
    for chunk in chunk_text(text):
        all_chunks.append((name, chunk))
print(f"Total chunks: {len(all_chunks)}")



# STORE

client  = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("nimbus_docs")

texts = [chunk for (name, chunk) in all_chunks]
ids = [f"chunk-{i}" for i in range(len(all_chunks))]
metadatas = [{"source": name} for (name, chunk) in all_chunks]


collection.upsert(documents=texts, ids=ids, metadatas=metadatas)

print(f"Stored {collection.count()} chunks in the vector database")



