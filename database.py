
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DATABASE_PATH


class Database:

    def __init__(self):
        self.database_path = DATABASE_PATH
        self.create_tables()

    # ======================================
    # DATABASE CONNECTION
    # ======================================

    @contextmanager
    def get_connection(self):
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=30
        )

        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    # ======================================
    # CREATE DATABASE TABLES
    # ======================================

    def create_tables(self):

        with self.get_connection() as connection:

            connection.execute("""
                CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_name TEXT NOT NULL,
                    input_text TEXT,
                    result TEXT,
                    confidence REAL,
                    created_at TEXT NOT NULL
                )
            """)

    # ======================================
    # TIMESTAMP
    # ======================================

    @staticmethod
    def current_timestamp():

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ======================================
    # ADD UPLOAD
    # ======================================

    def add_upload(self, filename, file_type):

        with self.get_connection() as connection:

            cursor = connection.execute("""
                INSERT INTO uploads (
                    filename,
                    file_type,
                    uploaded_at
                )
                VALUES (?, ?, ?)
            """, (
                filename,
                file_type,
                self.current_timestamp()
            ))

            return cursor.lastrowid

    # ======================================
    # ADD REGISTERED FACE
    # ======================================

    def add_face(self, person_name, image_path):

        with self.get_connection() as connection:

            cursor = connection.execute("""
                INSERT INTO faces (
                    person_name,
                    image_path,
                    created_at
                )
                VALUES (?, ?, ?)
            """, (
                person_name,
                str(image_path),
                self.current_timestamp()
            ))

            return cursor.lastrowid

    # ======================================
    # GET REGISTERED FACES
    # ======================================

    def get_faces(self):

        with self.get_connection() as connection:

            rows = connection.execute("""
                SELECT *
                FROM faces
                ORDER BY created_at DESC
            """).fetchall()

            return [
                dict(row)
                for row in rows
            ]

    # ======================================
    # SAVE ANALYSIS RESULT
    # ======================================

    def save_analysis(
        self,
        module_name,
        input_text,
        result,
        confidence=None
    ):

        with self.get_connection() as connection:

            cursor = connection.execute("""
                INSERT INTO analysis_results (
                    module_name,
                    input_text,
                    result,
                    confidence,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                module_name,
                str(input_text),
                str(result),
                confidence,
                self.current_timestamp()
            ))

            return cursor.lastrowid

    # ======================================
    # GET ANALYSIS HISTORY
    # ======================================

    def get_analysis_history(self, limit=50):

        limit = max(
            1,
            min(int(limit), 500)
        )

        with self.get_connection() as connection:

            rows = connection.execute("""
                SELECT *
                FROM analysis_results
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()

            return [
                dict(row)
                for row in rows
            ]

    # ======================================
    # DASHBOARD STATISTICS
    # ======================================

    def get_dashboard_stats(self):

        with self.get_connection() as connection:

            upload_count = connection.execute("""
                SELECT COUNT(*)
                FROM uploads
            """).fetchone()[0]

            face_count = connection.execute("""
                SELECT COUNT(DISTINCT person_name)
                FROM faces
            """).fetchone()[0]

            analysis_count = connection.execute("""
                SELECT COUNT(*)
                FROM analysis_results
            """).fetchone()[0]

            return {
                "total_uploads": upload_count,
                "registered_faces": face_count,
                "total_analyses": analysis_count
            }
