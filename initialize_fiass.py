import faiss
import json
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import os

# ----------------------------
# CONFIG
# ----------------------------
MONGO_URI = ""
DB_NAME = "interview_system"
COLLECTION_NAME = "interviews"

FAISS_INDEX_FILE = "faiss_index.bin"
FAISS_META_FILE = "faiss_docs.json"

# ----------------------------
# LOAD MODEL
# ----------------------------
print("📌 Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")
dim = 768

# ----------------------------
# LOAD OR CREATE FAISS
# ----------------------------
if os.path.exists(FAISS_INDEX_FILE):
    print("🔄 Loading existing FAISS index...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(FAISS_META_FILE, "r") as f:
        DOCUMENTS = json.load(f)
else:
    print("🆕 Creating new FAISS index...")
    index = faiss.IndexFlatL2(dim)
    DOCUMENTS = []

# ----------------------------
# CONNECT MONGO
# ----------------------------
print("📌 Connecting MongoDB...")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ----------------------------
# FETCH INTERVIEWS
# ----------------------------
interviews = list(collection.find({}, {"_id": 0}))
print(f"📊 Interviews found: {len(interviews)}")

# ----------------------------
# Chunk helper
# ----------------------------
def chunk_text(text, size=350):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ----------------------------
# PROCESS EACH INTERVIEW
# ----------------------------
added = 0

for interview in interviews:
    iid = interview.get("interview_id")
    email = interview.get("student_email")

    # Combine most important fields
    final_report = interview.get("final_report", {})
    conversation = interview.get("conversation_history", [])
    evaluations = interview.get("response_analyses", [])
    jd_match = interview.get("jd_match_data", {})

    text = json.dumps({
        "interview_id": iid,
        "email": email,
        "conversation": conversation,
        "evaluations": evaluations,
        "jd_match": jd_match,
        "final_report": final_report
    }, indent=2)

    chunks = chunk_text(text)

    for c in chunks:
        emb = model.encode([c])[0].astype("float32")
        index.add(np.array([emb]))

        DOCUMENTS.append({
            "interview_id": iid,
            "email": email,
            "chunk": c
        })

    added += len(chunks)
    print(f"✓ Added interview {iid} → {len(chunks)} chunks")

# ----------------------------
# SAVE INDEX + METADATA
# ----------------------------
faiss.write_index(index, FAISS_INDEX_FILE)
with open(FAISS_META_FILE, "w") as f:
    json.dump(DOCUMENTS, f, indent=2)

print("\n🎉 COMPLETED!")
print(f"🔢 Total chunks stored: {added}")
print("📂 Saved: faiss_index.bin + faiss_docs.json")
