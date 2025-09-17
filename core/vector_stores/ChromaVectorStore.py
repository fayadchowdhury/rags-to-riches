from core.vector_stores.BaseVectorStore import BaseVectorStore
import chromadb
import os
import uuid

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.persist_directory = self.config.get("persist_directory", "data/output/chroma_local")
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path = self.persist_directory
        )
        self.collection_name = self.config.get("collection_name", "")
        self.embedding_dim = self.config.get("embedding_dim", 1536)

        if not self.collection_name in [coll.name for coll in self.client.list_collections()]:
            self.collection = self.client.create_collection(
                name = self.collection_name,
            )
        else:
            self.collection = self.client.get_collection(self.collection_name)
    
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
        self.collection.add(
            ids=[d["id"] for d in self.prepared_data],
            embeddings=[d["embedding"] for d in self.prepared_data],
            metadatas=[d["metadata"] for d in self.prepared_data],
        )
        return 
    
    def store_batch(self, data, batch_size):
        self.store(data)
        return
    
    def query_top_k(self, query_embedding, k):
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["metadatas", "distances"]
        )

        self.query_docs = [
            {
                "metadata": metadata,
                "score": 1 - distance
            }
            for metadata, distance in zip(results["metadatas"][0], results["distances"][0])
        ]

        return self.query_docs