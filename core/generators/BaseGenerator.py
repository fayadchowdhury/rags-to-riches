from abc import ABC, abstractmethod
from typing import Iterator

class BaseGenerator(ABC):
    '''
    Abstract base class for generators
    '''

    def __init__(self, **kwargs):
        '''
        Initialize an LLM generator with optional configuration parameters
        '''
        self.config = kwargs

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        '''
        Take a query and a context to pass to LLM
        Return the output string
        '''
        pass

    @abstractmethod
    def generate_stream(self, query: str) -> Iterator[str]:
        '''
        Take a query to pass to LLM
        Return stream response as an iterator over a string
        '''
        pass