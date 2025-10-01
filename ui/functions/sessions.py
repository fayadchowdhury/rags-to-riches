from typing import List, Dict
from core.database.BaseDatabase import BaseDatabase

def get_session_from_db(db: BaseDatabase, session_id: str) -> Dict:
    '''
    Take a db object and a session ID string to fetch that session
    Return session as a dictionary
    '''
    session = db.get_session(session_id)
    return session

def get_sessions_from_db(db: BaseDatabase) -> List[Dict]:
    '''
    Take a db object to fetch all sessions
    Return a list of dictionary objects of each session
    '''
    sessions = db.get_all_sessions()
    if not sessions:
        return []
    return sessions

def create_new_session(name: str) -> Dict:
    '''
    Take a name string
    Return a new session as a dictionary
    '''
    return {
        "name": name,
        "named": 0,
        "messages": []
    }

def save_session_to_db(db: BaseDatabase, session: Dict) -> Dict:
    '''
    Take a db object and a session object dictionary to save to the database
    Return saved sesion as a dictionary
    '''
    session = db.save_session(session)
    return session

def get_chat_history(messages: List[Dict], token_limit: int) -> List[Dict]:
    '''
    Take a list of doctionary objects of messages and a token limit integer to get most recent chat history
    Return a list of dictionary objects of each message in chat history
    '''
    token_count = 0
    history = []
    for message in reversed(messages):
        content = message["content"]
        role = message["role"]
        created_at = message["created_at"]
        token_count += len(content.split())
        # Check to see if within token_limit
        if token_count > token_limit:
            break
        else:
            history.append({
                "role": role,
                "content": content,
                "created_at": created_at
            })
    return [c for c in reversed(history)]

def get_summary_from_db(db: BaseDatabase, session_id: str) -> Dict:
    '''
    Take a db object and a session ID string to get summary for the chat session
    Return summary as a dictionary
    '''
    summary = db.get_summary(session_id)
    return summary

def save_summary_to_db(db: BaseDatabase, session_id: str, summary: Dict) -> Dict:
    '''
    Take a db object, a session ID string and a summary object dictionary to save to the chat session in the database
    Return saved summary as a dictionary
    '''
    summary = db.save_summary(session_id, summary)
    return summary