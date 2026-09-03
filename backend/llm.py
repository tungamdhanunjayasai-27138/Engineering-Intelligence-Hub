from ollama import chat

MODEL = "llama3.2:3b"


def generate_answer(question, context):
    prompt = f"""
You are an engineering document intelligence assistant.

Answer the user's question using ONLY the provided document context.

Rules:
- Carefully understand the context before answering.
- Give a clear, useful and detailed answer.
- For broad questions, provide a structured explanation.
- For "What is this document about?" or similar questions, explain:
  1. The main topic
  2. The purpose of the work
  3. The main methods or approach
  4. The important findings or observations
  5. The main conclusions
- For specific questions, answer directly and explain the relevant details.
- Do not invent information.
- Do not use outside knowledge.
- If the provided context genuinely does not contain enough information, say so.
- Mention equations, experiments, simulations, figures, or technical methods when they are relevant.
- Prefer factual explanations over vague summaries.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]