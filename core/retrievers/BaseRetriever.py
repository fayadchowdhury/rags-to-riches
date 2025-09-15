from abc import ABC, abstractmethod
from typing import List, Dict

from core.embedders.BaseEmbedder import BaseEmbedder
from core.vector_stores.BaseVectorStore import BaseVectorStore

class BaseRetriever(ABC):
    '''
    Abstract base class for retrievers
    '''

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore, **kwargs):
        '''
        Initialize a retriever with optional configuration parameters
        '''
        self.config = kwargs
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str) -> List[Dict]:
        '''
        Take a query string
        Query vector store with embedded string and return list of dictionary objects
        '''
        pass