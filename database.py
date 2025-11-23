import sqlite3
from datetime import datetime

DB_NAME = "chat_history.db"

def init_db():
    """Initialize the database table"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_record(question, answer):
    """Create a new history record"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO history (timestamp, question, answer) VALUES (?, ?, ?)',
              (timestamp, question, answer))
    conn.commit()
    conn.close()

def get_all_records():
    """Read all history records, newest first"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY id DESC')
    records = c.fetchall()
    conn.close()
    return records

def update_record(record_id, question, answer):
    """Update an existing record"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('UPDATE history SET question = ?, answer = ? WHERE id = ?',
              (question, answer, record_id))
    conn.commit()
    conn.close()

def delete_record(record_id):
    """Delete a record"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM history WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
