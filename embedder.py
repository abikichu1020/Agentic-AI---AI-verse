from sentence_transformers import SentenceTransformer

# Load local embedding model (no API needed)
model = SentenceTransformer("all-MiniLM-L6-v2")   # 384 dims

def get_embedding(text: str):
    """
    Generate a dense vector embedding using SentenceTransformers.
    Returns a Python list of floats.
    """
    try:
        vector = model.encode(text)
        return vector.tolist()
    except Exception as e:
        print("❌ Embedding generation error:", e)
        return None
