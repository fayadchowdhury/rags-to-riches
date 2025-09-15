from core.retrievers.BaseRetriever import BaseRetriever

class TopKRetriever(BaseRetriever):
    def __init__(self, embedder, vector_store, **kwargs):
        super().__init__(embedder, vector_store, **kwargs)
        self.k = self.config.get("top_k", 10)

    def retrieve(self, query):
        return_docs = []
        self.query_embedding = self.embedder.embed_text(query)
        query_docs = self.vector_store.query_top_k(self.query_embedding, self.k)
        for doc in query_docs["matches"]:
            text = doc["metadata"]["text"]
            return_docs.append(" ".join(text))

        return return_docs
