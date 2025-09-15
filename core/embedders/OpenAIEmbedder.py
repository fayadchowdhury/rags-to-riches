from core.embedders.BaseEmbedder import BaseEmbedder
from openai import OpenAI


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs
        self.client = OpenAI(
            api_key = self.config.get("api_key", "")
        )
        self.model = self.config.get("embedding_model", "text-embedding-ada-002")

    def embed_text(self, text):
        return self.client.embeddings.create(
                    model=self.model,
                    input=text
                ).data[0].embedding

    def embed_data(self, data):
        self.embeddings = []
        for item in data:
            item["embedding"] = self.client.embeddings.create(
                model=self.model,
                input=item["text"]
            ).data[0].embedding
            self.embeddings.append(item)
        return self.embeddings