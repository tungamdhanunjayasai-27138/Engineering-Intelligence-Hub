import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-3.6-flash"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_answer(question, context):

    prompt = f"""
You are an engineering document assistant.

Answer the user's question using ONLY the provided document context.

Rules:
- Use only information present in the context.
- Do not use outside knowledge.
- Do not invent or assume facts.
- Answer directly and clearly.
- If the context does not contain enough information to answer,
  say exactly:
  "The document does not contain enough information."
- If the question asks for multiple things, answer each part
  that is supported by the context.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text


def generate_document_summary(context):

    prompt = f"""
You are an engineering document assistant.

Analyze the provided document and give a concise summary.

Include:
- Main topic
- Objective
- Methodology
- Important findings
- Conclusion

Use ONLY the provided document.
Do not invent information.

DOCUMENT:
{context}

SUMMARY:
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text


def generate_web_answer(question, document_context=""):

    prompt = f"""
You are an engineering research assistant.

Answer the user's question accurately.

DOCUMENT:
{document_context}

QUESTION:
{question}

Clearly distinguish information from the document from additional
information.

Do not invent information.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text