from core.database.BaseDatabase import BaseDatabase

def get_messages_for_session_from_db(db: BaseDatabase, session_id: str):
    messages = db.get_messages(session_id)
    if not messages:
        return []
    return messages

def save_message_for_session_to_db(db: BaseDatabase, session_id: str, message: dict):
    message = db.save_message(session_id, message)
    return message

def delete_messages_for_session_from_db(db: BaseDatabase, session_id: str):
    db.delete_messages(session_id)
    return