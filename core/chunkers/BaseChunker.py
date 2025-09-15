from abc import ABC, abstractmethod
from typing import List, Dict

class BaseChunker(ABC):
    '''
    Abstract base class for chunkers
    '''
    
    def __init__(self, **kwargs):
        '''
        Initialize a chunker with optional configuration parameters
        '''
        self.config = kwargs

    @abstractmethod
    def chunk(self, data: List[Dict]) -> List[Dict]:
        '''
        Take a list of dictionaries structured text and metadata
        Return a list of dictionaries of chunks of structured text and metadata
        '''
        pass