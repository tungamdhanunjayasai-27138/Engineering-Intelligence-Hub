from ollama import chat

MODEL = "llama3.2:3b"


def ask_llm(prompt):
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def generate_section_summary(text):
    prompt = f"""
You are extracting factual information from a scientific document.

Use ONLY the text provided.

Extract:
- What is being studied
- Why it is being studied
- Methods/models/simulations used
- Important technical concepts
- Important numerical results
- Findings and observations
- Conclusions
- Limitations or future work

STRICT RULES:
- Do not guess.
- Do not use general knowledge.
- Do not identify the document type unless explicitly stated.
- Do not introduce subjects, objects, systems or applications
  that are not explicitly present.
- Preserve important names, terminology, values and equations.
- If information is not present, omit it.

TEXT:
{text}

FACTUAL SUMMARY:
"""
    return ask_llm(prompt)


def generate_document_summary(text):
    words = text.split()
    chunk_size = 1200

    summaries = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        summaries.append(generate_section_summary(chunk))

    combined = "\n\n--- SECTION ---\n\n".join(summaries)

    prompt = f"""
You are creating a factual overview of a scientific document.

Use ONLY the extracted summaries below.

Create an answer with these sections:

## Main Topic
## Objective
## Approach / Methodology
## Important Concepts
## Key Findings
## Conclusion
## Limitations / Future Work

STRICT RULES:
- Every statement must be supported by the summaries.
- Do NOT use outside knowledge.
- Do NOT guess what the document is.
- Do NOT call it a thesis, research paper, report, etc.
  unless explicitly supported.
- Do NOT introduce AGN, stars, CVs, YSOs, or any other
  specific system unless supported by the summaries.
- If a section is not supported, write:
  "Not clearly stated in the document."
- Prefer factual details over generic descriptions.
- Preserve technical terminology and numerical values.
- Do not mention summaries, chunks, context, embeddings or retrieval.

EXTRACTED SUMMARIES:
{combined}

FINAL ANSWER:
"""

    return ask_llm(prompt)


def generate_answer(question, context):
    prompt = f"""
You are an engineering/scientific document assistant.

Answer the question using ONLY the document text below.

STRICT RULES:
- Do not use outside knowledge.
- Do not invent facts.
- Do not guess missing information.
- Use the terminology used in the document.
- Combine relevant information when necessary.
- If the answer is not supported, say:
  "The document does not provide enough information to answer this."

DOCUMENT:
{context}

QUESTION:
{question}

ANSWER:
"""

    return ask_llm(prompt)