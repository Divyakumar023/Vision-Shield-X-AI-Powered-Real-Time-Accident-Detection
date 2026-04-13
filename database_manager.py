import sqlite3
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self, db_path="accident_history.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    location TEXT NOT NULL,
                    description TEXT,
                    image_path TEXT,
                    is_verified INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def log_incident(self, camera_id, location, description, image_path, is_verified):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incidents (timestamp, camera_id, location, description, image_path, is_verified)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, camera_id, location, description, image_path, 1 if is_verified else 0))
            conn.commit()
            return cursor.lastrowid

    def get_recent_incidents(self, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,))
            return cursor.fetchall()
