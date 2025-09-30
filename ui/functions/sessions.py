from typing import List, Dict
from core.database.BaseDatabase import BaseDatabase

def get_session_from_db(db: BaseDatabase, session_id: str):
    session = db.get_session(session_id)
    return session

def get_sessions_from_db(db: BaseDatabase):
    sessions = db.get_all_sessions()
    if not sessions:
        return []
    return sessions

def create_new_session(name: str):
    return {
        "name": name,
        "messages": []
    }

def save_session_to_db(db: BaseDatabase, session: Dict):
    session = db.save_session(session)
    return session

def get_chat_history(messages: List[Dict], token_limit: int):
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
    summary = db.get_summary(session_id)
    return summary

def save_summary_to_db(db: BaseDatabase, session_id: str, summary: Dict):
    summary = db.save_summary(session_id, summary)
    return summary