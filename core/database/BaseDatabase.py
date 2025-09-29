from abc import ABC, abstractmethod
from typing import List, Dict

class BaseDatabase(ABC):
    def __init__(self, **kwargs):
        '''
        Initialize a database with optional configuration parameters
        '''
        self.config = kwargs

    @abstractmethod
    def _create_tables(self):
        '''
        Create necessary tables in the database
        '''
        pass

    @abstractmethod
    def save_session(self, session: Dict):
        '''
        Save a chat session with its metadata in the database
        '''
        pass

    @abstractmethod
    def get_all_sessions(self) -> List[Dict]:
        '''
        Retrieve all chat sessions from the database
        '''
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Dict:
        '''
        Retrieve a chat session by its ID
        '''
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str):
        '''
        Delete a chat session by its ID
        '''
        pass

    @abstractmethod
    def save_message(self, session_id: str, message: Dict):
        '''
        Save a message in a chat session
        '''
        pass

    @abstractmethod
    def get_messages(self, session_id: str) -> List[Dict]:
        '''
        Retrieve all messages for a chat session
        '''
        pass

    @abstractmethod
    def delete_messages(self, session_id: str):
        '''
        Delete all messages for a chat session
        '''
        pass

    @abstractmethod
    def save_summary(self, session_id: str, summary: Dict):
        '''
        Save summmary of a chat session
        '''
        pass

    @abstractmethod
    def get_summary(self, session_id: str) -> Dict:
        '''
        Retrieve summary of a chat session
        '''
        pass

    @abstractmethod
    def close(self):
        '''
        Close the database connection
        '''
        pass