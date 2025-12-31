import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

FAISS_PATH = "faiss_index.bin"
META_PATH = "faiss_docs.json"

# -------------------------------
# Load or create FAISS index
# -------------------------------
if os.path.exists(FAISS_PATH):
    FAISS_INDEX = faiss.read_index(FAISS_PATH)
    print("📌 Loaded existing FAISS index")
else:
    FAISS_INDEX = faiss.IndexFlatL2(768)
    print("🆕 Created new FAISS index")

# Load metadata
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        DOCUMENTS = json.load(f)
else:
    DOCUMENTS = []

# -------------------------------
# Save FAISS + metadata
# -------------------------------
def save_faiss():
    faiss.write_index(FAISS_INDEX, FAISS_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(DOCUMENTS, f, indent=2)
    print("💾 FAISS + metadata saved")


# -------------------------------
# Add interview to FAISS
# -------------------------------
def add_interview_to_faiss(interview_id, email, text):
    chunks = split_into_chunks(text, chunk_size=350)

    for chunk in chunks:
        embedding = model.encode([chunk])[0]
        FAISS_INDEX.add(np.array([embedding]).astype("float32"))

        DOCUMENTS.append({
            "interview_id": interview_id,
            "email": email,
            "chunk": chunk
        })

    save_faiss()
    print(f"🔍 Added Interview {interview_id} with {len(chunks)} chunks")


# -------------------------------
# Split text into meaningful chunks
# -------------------------------
def split_into_chunks(text, chunk_size=350):
    words = text.split()
    return [
        " ".join(words[i: i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# -------------------------------
# Search FAISS
# -------------------------------
def search_faiss(query, top_k=5):
    if len(DOCUMENTS) == 0:
        return []

    embedding = model.encode([query])[0]
    embedding = np.array([embedding]).astype("float32")

    distances, indices = FAISS_INDEX.search(embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(DOCUMENTS):
            results.append(DOCUMENTS[idx])

    return results
