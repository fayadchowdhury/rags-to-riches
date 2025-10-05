from typing import List, Dict, Any

from core.pipelines.BasePipeline import BasePipeline
from core.utils.pipeline_utils import parser_router

class UIPipeline(BasePipeline):
    def __init__(self, parsers, chunker, embedder, vector_store, retriever, generator, **kwargs):
        super().__init__(parsers, chunker, embedder, vector_store, retriever, generator, **kwargs)
        self.summary_prompt_template = self.config["summary_prompt_template"]
        self.title_prompt_template = self.config["title_prompt_template"]
        self.query_prompt_template = self.config["query_prompt_template"]

    def ingest_object(self, obj):
        parser = parser_router(self.parsers, obj)
        parsed_document = parser.parse(obj)
        chunks = self.chunker.chunk(parsed_document)
        embeddings = self.embedder.embed_data(chunks)
        self.vector_store.store_batch(embeddings, batch_size=10)
    
    def ingest_objects_from_directory(self, directory):
        return super().ingest_objects_from_directory(directory)
    
    def query(self, query, history):
        context_names, context = self.retriever.retrieve(query)
        query_with_context_injected = self.query_prompt_template.format(query=query, context=context)
        context = [
            {
                "title": title,
                "content": content
            }
            for title, content in zip(context_names, context)
        ]
        return context, self.generator.generate_stream(history, query_with_context_injected)
    
    def update_component(self, component_name: str, new_component: Any):
        '''
        Takes a compoennt name string and the new component to replace it with
        '''
        setattr(self, component_name, new_component)

    def list_components(self) -> Dict:
        '''
        Return a dictionary of current components in pipeline
        '''
        return {
            "parsers": self.parsers,
            "chunker": self.chunker,
            "embedder": self.embedder,
            "vector_store": self.vector_store,
            "retriever": self.retriever,
            "generator": self.generator,
        }
    
    def reset(self, **new_config):
        '''
        Takes a config dictionary to reset pipeline to
        '''
        self.config.update(new_config)

    def summarize_session(self, messages: List[Dict]) -> str:
        '''
        Takes a list of messages and generates summary
        Returns summary as a string
        '''
        query = ""
        for message in messages:
            query += f"{message["role"]}: {message["content"]}\n\n"

        query += self.summary_prompt_template
        return self.generator.generate(query, "")
    
    def generate_title(self, prompt: str) -> str:
        '''
        Takes an input prompt to base the title of the session on
        Returns title as a string
        '''
        query = self.title_prompt_template + "\n\n" +  prompt
        return self.generator.generate(query, "")
