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

def save_session_to_db(db: BaseDatabase, session: dict):
    session_id = db.save_session(session)
    return session_id