from core.vector_stores.BaseVectorStore import BaseVectorStore
from pinecone import Pinecone, ServerlessSpec
import uuid

class PineconeVectorStore(BaseVectorStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Pinecone(
            api_key=self.config.get("api_key", "")
        )
        self.index_name = self.config.get("index_name", "")
        self.embedding_dim = self.config.get("embedding_dim", 1536)
        if not self.client.has_index(self.index_name):
            self.client.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.client.Index(self.index_name)
    
    def _prepare_data_for_upsert(self, data):
        prepared_data = []
        for item in data:
            embedding = list(item['embedding'])  # Ensure embedding is in list format
            # Prepare the metadata dictionary with the relevant columns
            metadata = {
                "file_type": item['file_type'],
                "file_name": item['file_name'],
                "marker": str(item['marker']),
                "sub_marker": str(item['sub_marker']),
                "first_10_tokens": item['first_10_tokens'],
                "text": item['text']
            }
            prepared_data.append({
                'id': str(uuid.uuid4()),
                'values': embedding,
                'metadata': metadata
            })
        
        return prepared_data

    def store(self, data):
        self.prepared_data = self._prepare_data_for_upsert(data)
        self.index.upsert(vectors=self.prepared_data)
        return 
    
    def store_batch(self, data, batch_size):
        self.prepared_data = self._prepare_data_for_upsert(data)
        for i in range(0, len(self.prepared_data), batch_size):
            batch = self.prepared_data[i:i + batch_size]
            self.index.upsert(vectors=batch)
        return
    
    def query_top_k(self, query_embedding, k):
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            include_values=False
        )

        self.query_docs = [
            {
                "metadata": match["metadata"],
                "score": match["score"]
            } for match in results["matches"]
        ]

        return self.query_docs