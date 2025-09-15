from abc import ABC
from typing import Any, List

from core.parsers.BaseParser import BaseParser
from core.chunkers.BaseChunker import BaseChunker
from core.embedders.BaseEmbedder import BaseEmbedder
from core.vector_stores.BaseVectorStore import BaseVectorStore
from core.retrievers.BaseRetriever import BaseRetriever
from core.generators.BaseGenerator import BaseGenerator

class BasePipeline(ABC):
    '''
    Abstract base class for RAG pipelines
    '''

    def __init__(self, parsers: List[BaseParser], chunker: BaseChunker, embedder: BaseEmbedder, vector_store: BaseVectorStore, retriever: BaseRetriever, generator: BaseGenerator, **kwargs):
        '''
        Initialize a RAG pipeline with
        - parser
        - chunker
        - embedder
        - vector_store
        - retriever
        - generator
        and optional configuration parameters
        '''
        self.parsers = parsers
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator
        self.config = kwargs

    def ingest_object(self, obj: Any):
        '''
        Ingest a single object into pipeline
        '''
        pass

    def ingest_objects_from_directory(self, directory: str):
        '''
        Ingest all objects in directory into pipeline
        '''
        pass

    def query(self, query: str) -> str:
        '''
        Take a query string
        Return the output string
        '''
        pass