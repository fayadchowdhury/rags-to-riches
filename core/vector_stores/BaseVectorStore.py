from abc import ABC, abstractmethod
from typing import Dict, List

class BaseVectorStore(ABC):
    '''
    Abstract base class for vector stores
    '''

    def __init__(self, **kwargs):
        '''
        Initialize a vectore store with optional configuration parameters and embedder
        '''
        self.config = kwargs

    @abstractmethod
    def store(self, data: Dict):
        '''
        Take a dictionary of embeddings and metadata
        Store in vector store
        '''
        pass

    @abstractmethod
    def store_batch(self, data: Dict, batch_size: int):
        '''
        Take a dictionary of embeddings and metadata
        Store in vector store in batches
        '''
        pass

    @abstractmethod
    def query_top_k(self, query_embedding: List[float], k: int) -> List[Dict]:
        '''
        Take a query embedding
        Return a list of dictionary objects of retrieved documents
        '''
        pass