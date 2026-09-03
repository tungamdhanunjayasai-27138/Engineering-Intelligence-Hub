from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

client = QdrantClient(path="data/qdrant")

COLLECTION_NAME = "engineering_docs"

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )


def store_documents(documents):
    points = []

    for document in documents:
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=document["embedding"],
                payload={
                    "source": document["source"],
                    "chunk_id": document["chunk_id"],
                    "text": document["text"]
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


def search_documents(query_vector, limit=10):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    )

    return [
        {
            "source": point.payload["source"],
            "text": point.payload["text"],
            "score": point.score
        }
        for point in results.points
    ]