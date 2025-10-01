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
    def save_session(self, session: Dict) -> Dict:
        '''
        Take a session object and save it to the database
        Return saved session as a dictionary
        '''
        pass

    @abstractmethod
    def get_all_sessions(self) -> List[Dict]:
        '''
        Retrieve all chat sessions from the database
        Return a list of dictionary objects of each session
        '''
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Dict:
        '''
        Take a session ID string and retrieve a chat session by its ID
        Return session as a dictionary
        '''
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str):
        '''
        Take a session ID string and delete a chat session by its ID
        '''
        pass

    @abstractmethod
    def save_message(self, session_id: str, message: Dict) -> Dict:
        '''
        Take a session ID string and a message dictionary object and save it to the chat session in the database
        Return saved message as a dictionary
        '''
        pass

    @abstractmethod
    def get_messages(self, session_id: str) -> List[Dict]:
        '''
        Take a session ID string and retrieve all messages for a chat session by ID
        Return a list of dictionary objects of each message
        '''
        pass

    @abstractmethod
    def delete_messages(self, session_id: str):
        '''
        Take a session ID string and delete messages for that session from the database
        '''
        pass

    @abstractmethod
    def save_summary(self, session_id: str, summary: Dict) -> Dict:
        '''
        Take a session ID string and a summary object and save it to the chat session in the database
        Return saved summary as a dictionary
        '''
        pass

    @abstractmethod
    def get_summary(self, session_id: str) -> Dict:
        '''
        Take a session ID string and retrieve summary of a chat session
        Return summary as a dictionary
        '''
        pass

    @abstractmethod
    def close(self):
        '''
        Close the database connection
        '''
        pass