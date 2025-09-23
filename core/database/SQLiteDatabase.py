from core.database.BaseDatabase import BaseDatabase
import sqlite3
import uuid
import os

class SQLiteDatabase(BaseDatabase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.db_path = self.config.get("DB_DATABASE", "database.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
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
        self.conn.commit()
    
    def save_session(self, session):
        self.cursor.execute('''
            INSERT OR REPLACE INTO sessions (id, name, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (str(uuid.uuid4()), session["name"])
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_all_sessions(self):
        self.cursor.execute('''
            SELECT id, name, created_at FROM sessions
            ORDER BY created_at DESC
        ''')
        rows = self.cursor.fetchall()
        sessions = [dict(session) for session in rows]
        return sessions

    def get_session(self, session_id):
        self.cursor.execute('''
            SELECT id, name, created_at FROM sessions
            WHERE id = ?
        ''', (session_id,))
        row = self.cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def delete_session(self, session_id):
        self.cursor.execute('''
            DELETE FROM sessions WHERE id = ?
        ''', (session_id,))
        self.conn.commit()
        return self.cursor.rowcount
    
    def save_message(self, session_id, message):
        self.cursor.execute('''
            INSERT INTO messages (id, session_id, role, content)
            VALUES (?, ?, ?, ?)
        ''', (str(uuid.uuid4()), session_id, message["role"], message["content"])
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_messages(self, session_id):
        self.cursor.execute('''
            SELECT id, role, content, created_at FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
        ''', (session_id,))
        rows = self.cursor.fetchall()
        messages = [dict(message) for message in rows]
        return messages
    
    def delete_messages(self, session_id):
        self.cursor.execute('''
            DELETE FROM messages WHERE session_id = ?
        ''', (session_id,))
        self.conn.commit()
        return self.cursor.rowcount

    def close(self):
        self.conn.close()
        return super().close()