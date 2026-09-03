from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.ingestion import load_documents, process_documents
from backend.vector_store import store_documents, search_documents
from backend.vector_store import client, COLLECTION_NAME
from backend.embeddings import create_embedding
from backend.llm import generate_answer

app = FastAPI(title="Engineering Intelligence Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = (
        ".pdf", ".docx", ".txt", ".md",
        ".csv", ".xlsx", ".pptx", ".html", ".htm"
    )

    if not file.filename.lower().endswith(allowed):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type."
        )

    file_path = DOCUMENTS_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    return {
        "message": "Document uploaded successfully",
        "filename": file.filename
    }

@app.get("/health")
def health():
    collections = client.get_collections()

    qdrant_ready = any(
        c.name == COLLECTION_NAME
        for c in collections.collections
    )

    return {
        "status": "healthy",
        "service": "engineering-intelligence-hub",
        "qdrant": "ready" if qdrant_ready else "not_ready"
    }

@app.get("/")
def home():
    return {"message": "Engineering Intelligence Hub is running!"}


@app.get("/documents")
def documents():
    return load_documents()


@app.get("/process")
def process():
    return process_documents()


@app.post("/index")
def index_documents():
    documents = process_documents()
    store_documents(documents)

    return {
        "message": "Documents indexed successfully",
        "chunks_indexed": len(documents)
    }

@app.get("/search")
def search(query: str):
    vector = create_embedding(query)
    return search_documents(vector)


@app.get("/ask")
def ask(query: str):
    try:
        vector = create_embedding(query)
        results = search_documents(vector, limit=10)

        if not results:
            return {
                "question": query,
                "answer": "I don't know. No relevant information was found.",
                "sources": []
            }

        context = "\n\n".join(
            result["text"] for result in results
        )

        answer = generate_answer(query, context)

        sources = [
            {
                "source": result["source"],
                "score": round(result["score"], 3)
            }
            for result in results
        ]

        return {
            "question": query,
            "answer": answer,
            "sources": sources
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to process the question."
        )