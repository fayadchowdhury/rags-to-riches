from abc import ABC, abstractmethod
from typing import Any, List, Dict

class BaseParser(ABC):
    '''
    Abstract base class for parsers
    '''
    
    def __init__(self, **kwargs):
        '''
        Initialize a parser with optional configuration parameters
        '''
        self.config = kwargs

    @abstractmethod
    def read(self, obj: Any) -> Any:
        '''
        Read an input object (file path, bytes, stream etc.)
        Return a raw intermediate representation
        '''
        pass

    @abstractmethod
    def parse(self, obj: Any) -> List[Dict]:
        '''
        Take an intermediate object
        Return a list of dictionaries of structured text and metadata
        '''
        pass