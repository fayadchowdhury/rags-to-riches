from abc import ABC, abstractmethod
from typing import Dict, List

class BaseEmbedder(ABC):
    '''
    Abstract base class for embedders
    '''

    def __init__(self, **kwargs):
        '''
        Initialize an embedder with optional configuration parameters
        '''
        self.config = kwargs

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        '''
        Take a text string
        Return a list of floats for the embedding vector
        '''
        pass

    @abstractmethod
    def embed_data(self, data: List[Dict]) -> List[Dict]:
        '''
        Take a dictionary of chunks and associated metadata
        Return chunks with embedding
        '''
        pass