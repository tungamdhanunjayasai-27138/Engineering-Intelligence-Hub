from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.vector_store import (
    clear_document,
    store_documents,
    search_documents,
    client,
    COLLECTION_NAME
)
from backend.ingestion import load_documents, process_documents
from backend.embeddings import create_embedding
from backend.llm import generate_answer, generate_document_summary


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

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md",
    ".html", ".htm", ".csv",
    ".xlsx", ".pptx"
}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type."
        )

    # Clear previous document from vector database
    for old_file in DOCUMENTS_DIR.iterdir():
        if old_file.is_file():
            clear_document(old_file.name)

            try:
                old_file.unlink()
            except PermissionError:
                pass

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
    return {
        "message": "Engineering Intelligence Hub is running!"
    }


@app.get("/documents")
def documents():
    return load_documents()


@app.get("/process")
def process():
    return process_documents()


@app.post("/index")
def index_documents():

    documents = process_documents()

    if documents:
        clear_document(documents[0]["source"])

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
        documents = load_documents()

        if not documents:
            return {
                "question": query,
                "answer": "No document is currently uploaded.",
                "sources": []
            }

        vector = create_embedding(query)

        results = search_documents(
            vector,
            limit=10,
            min_score=0
        )

        if not results:
            return {
                "question": query,
                "answer": "The document does not contain enough information.",
                "sources": []
            }

        context = "\n\n".join(
            r["text"] for r in results
        )

        answer = generate_answer(
            query,
            context
        )

        sources = [
            {
                "source": r["source"],
                "score": round(r["score"], 3)
            }
            for r in results
        ]

        return {
            "question": query,
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        print("ERROR:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )