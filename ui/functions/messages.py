from core.database.BaseDatabase import BaseDatabase
from typing import List, Dict

def get_messages_for_session_from_db(db: BaseDatabase, session_id: str) -> List[Dict]:
    '''
    Take a db object and a session ID string to fetch messages for that session from the database
    Return a list of dictionary objects of each message  
    '''
    messages = db.get_messages(session_id)
    if not messages:
        return []
    return messages

def save_message_for_session_to_db(db: BaseDatabase, session_id: str, message: dict) -> Dict:
    '''
    Take a db object, a session ID string and a message object dictionary to save message to the chat session in the database
    Return saved message as a dictionary  
    '''
    message = db.save_message(session_id, message)
    return message

def delete_messages_for_session_from_db(db: BaseDatabase, session_id: str):
    '''
    Take a db object and a session ID string to delete messages for that session from the database
    '''
    db.delete_messages(session_id)
    return