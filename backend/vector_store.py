from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

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


def clear_document(filename):
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=filename)
                )
            ]
        )
    )


def store_documents(documents):
    points = []

    for document in documents:
        point_id = abs(hash(
            document["source"] + str(document["chunk_id"])
        )) % (2**63)

        points.append(
            PointStruct(
                id=point_id,
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


def search_documents(query_vector, limit=10, min_score=0.05):
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
        if point.score >= min_score
    ]