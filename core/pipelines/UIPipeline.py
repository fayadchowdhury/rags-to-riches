from typing import List, Dict, Any

from core.pipelines.BasePipeline import BasePipeline

class UIPipeline(BasePipeline):
    def __init__(self, parsers, chunker, embedder, vector_store, retriever, generator, **kwargs):
        super().__init__(parsers, chunker, embedder, vector_store, retriever, generator, **kwargs)

    def ingest_object(self, obj):
        return super().ingest_object(obj)
    
    def ingest_objects_from_directory(self, directory):
        return super().ingest_objects_from_directory(directory)
    
    def query(self, prompt_template, query, history):
        context_names, context = self.retriever.retrieve(query)
        query_with_context_injected = prompt_template.format(query=query, context=context)
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

    def summarize_session(self, messages: List[Dict]) -> Dict:
        '''
        Take a list of messages
        Return a summary
        '''
        query = ""
        for message in messages:
            query += f"{message["role"]}: {message["content"]}\n\n"

        query += "Summarize the conversation thus far in less than 300 words, keeping track of what the user said and what the assistant responded with"
        return self.generator.generate(query, "")
