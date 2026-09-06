from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import uuid

from app.config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
)


class QdrantVectorStore:

    COLLECTION_NAME = "samsung_documents"

    VECTOR_SIZE = 384

    def __init__(self):

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    def upsert_chunk(
        self,
        chunk_id: str,
        vector: list[float],
        payload: dict,
    ):

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                chunk_id,
            )
        )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,

            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def create_collection(self):

        collections = self.client.get_collections()

        existing_collections = [
            collection.name
            for collection in collections.collections
        ]

        if self.COLLECTION_NAME in existing_collections:

            print(
                f"Collection already exists: "
                f"{self.COLLECTION_NAME}"
            )

            return

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,

            vectors_config=VectorParams(
                size=self.VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created collection: "
            f"{self.COLLECTION_NAME}"
        )
    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ):
        results = self.client.query_points(
        collection_name=self.COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
        )

        return results.points

    def test_connection(self):

        return self.client.get_collections()