def chunk_text(text, size=1000, overlap=150):
    text = " ".join(text.split())
    chunks = []

    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start:
                end = cut + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks