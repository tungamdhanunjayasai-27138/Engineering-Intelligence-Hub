def chunk_text(text, size=500, overlap=50):
    sections = text.split("\n## ")
    chunks = []

    for i, section in enumerate(sections):
        if i > 0:
            section = "## " + section

        section = section.strip()

        if len(section) <= size:
            chunks.append(section)
            continue

        start = 0

        while start < len(section):
            end = min(start + size, len(section))
            chunk = section[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += size - overlap

    return chunks