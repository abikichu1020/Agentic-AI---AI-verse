def chunk_text(text: str, max_words=120):
    """
    Splits long text into overlapping chunks for better RAG results.
    """
    words = text.split()
    chunks = []
    step = max_words - 40  # 40-word overlap for better context

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks
