from core.database.BaseDatabase import BaseDatabase
import sqlite3
import uuid
import os

class SQLiteDatabase(BaseDatabase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.db_path = self.config.get("DB_DATABASE", "database.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def save_session(self, session):
        session_id = session.get("id", str(uuid.uuid4()))
        self.cursor.execute('''
            INSERT OR REPLACE INTO sessions (id, name, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            RETURNING *
        ''', (session_id, session["name"])
        )
        row = self.cursor.fetchone()
        self.conn.commit()
        if row:
            session = {
                "id": row[0],
                "name": row[1],
                "created_at": row[2],
                "updated_at": row[3]
            }
            return session
        return None
    
    def get_all_sessions(self):
        self.cursor.execute('''
            SELECT id, name, created_at FROM sessions
            ORDER BY created_at DESC
        ''')
        rows = self.cursor.fetchall()
        sessions = [{
            "id": session[0],
            "name": session[1],
            "created_at": session[2]
        } for session in rows]
        return sessions

    def get_session(self, session_id):
        self.cursor.execute('''
            SELECT id, name, created_at FROM sessions
            WHERE id = ?
        ''', (session_id,))
        row = self.cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "created_at": row[2]
            }
        return None
    
    def delete_session(self, session_id):
        self.cursor.execute('''
            DELETE FROM sessions WHERE id = ?
        ''', (session_id,))
        self.conn.commit()
        return self.cursor.rowcount
    
    def save_message(self, session_id, message):
        message_id = message.get("id", str(uuid.uuid4()))
        self.cursor.execute('''
            INSERT INTO messages (id, session_id, role, content)
            VALUES (?, ?, ?, ?)
            RETURNING *
        ''', (message_id, session_id, message["role"], message["content"])
        )
        row = self.cursor.fetchone()
        self.conn.commit()
        if row:
            message = {
                "id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "created_at": row[4]
            }
            return message
        return None
    
    def get_messages(self, session_id):
        self.cursor.execute('''
            SELECT id, session_id, role, content, created_at FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
        ''', (session_id,))
        rows = self.cursor.fetchall()
        messages = [{
            "id": message[0],
            "session_id": message[1],
            "role": message[2],
            "content": message[3],
            "created_at": message[4]
        } for message in rows]
        return messages
    
    def delete_messages(self, session_id):
        self.cursor.execute('''
            DELETE FROM messages WHERE session_id = ?
        ''', (session_id,))
        self.conn.commit()
        return self.cursor.rowcount
    
    def save_summary(self, session_id, summary):
        summary_id = summary.get("id", str(uuid.uuid4()))
        self.cursor.execute('''
            INSERT INTO summaries (id, session_id, summary)
            VALUES (?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                session_id = excluded.session_id,
                summary = excluded.summary,
                updated_at = CURRENT_TIMESTAMP
            RETURNING *
        ''', (summary_id, session_id, summary["text"]))
        row = self.cursor.fetchone()
        self.conn.commit()
        if row:
            summary = {
                "id": row[0],
                "session_id": row[1],
                "summary": row[2],
                "created_at": row[3],
                "updated_at": row[4]
            }
            return summary
        return None

    def get_summary(self, session_id):
        self.cursor.execute('''
            SELECT id, session_id, summary, created_at, updated_at
            FROM summaries WHERE session_id = ?
        ''', (session_id,))
        row =  self.cursor.fetchone()
        if row:
            summary = {
                "id": row[0],
                "session_id": row[1],
                "summary": row[2],
                "created_at": row[3],
                "updated_at": row[4],
            }
            return summary
        return None

    def close(self):
        self.conn.close()
        return super().close()