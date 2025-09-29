from typing import Dict, Any

from core.pipelines.BasePipeline import BasePipeline

class UIPipeline(BasePipeline):
    def __init__(self, parsers, chunker, embedder, vector_store, retriever, generator, **kwargs):
        super().__init__(parsers, chunker, embedder, vector_store, retriever, generator, **kwargs)

    def ingest_object(self, obj):
        return super().ingest_object(obj)
    
    def ingest_objects_from_directory(self, directory):
        return super().ingest_objects_from_directory(directory)
    
    def query(self, prompt_template, query, history):
        context = self.retriever.retrieve(query)
        query_with_context_injected = prompt_template.format(query=query, context=context)
        return self.generator.generate_stream(history, query_with_context_injected)
    
    def update_component(self, component_name: str, new_component: Any):
        '''
        Replace a component in the pipeline from the UI at runtime
        Takes a compoennt name and the new component to replace it with
        '''
        setattr(self, component_name, new_component)

    def list_components(self) -> Dict:
        '''
        List current components in pipeline
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
        Reset pipeline to a new config
        Takes a config dictionary
        '''
        self.config.update(new_config)