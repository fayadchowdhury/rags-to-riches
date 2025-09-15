from abc import ABC, abstractmethod

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