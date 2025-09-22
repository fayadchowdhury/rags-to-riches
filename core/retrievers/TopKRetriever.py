from core.retrievers.BaseRetriever import BaseRetriever

class TopKRetriever(BaseRetriever):
    def __init__(self, embedder, vector_store, **kwargs):
        super().__init__(embedder, vector_store, **kwargs)
        self.k = self.config.get("top_k", 10)

    def retrieve(self, query):
        return_doc_names = []
        return_doc_texts = []
        self.query_embedding = self.embedder.embed_text(query)
        query_docs = self.vector_store.query_top_k(self.query_embedding, self.k)
        for doc in query_docs:
            text = doc["metadata"]["text"]
            doc_name = doc["metadata"]["file_name"]
            return_doc_names.append(doc_name)
            return_doc_texts.append(text)

        return return_doc_names, return_doc_texts
