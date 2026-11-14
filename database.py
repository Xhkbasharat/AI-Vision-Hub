
import logging
import sqlite3

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import DATABASE_PATH


logger = logging.getLogger("AI-Vision-Hub")


class Database:
    """
    SQLite database manager for AI Vision Hub.

    Stores:
    - Uploaded files
    - Registered faces
    - AI analysis results
    """

    def __init__(self, database_path=None):

        self.database_path = Path(
            database_path or DATABASE_PATH
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

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

        try:

            yield connection

            connection.commit()

        except Exception:

            connection.rollback()

            raise

        finally:

            connection.close()

    # ======================================
    # CURRENT TIMESTAMP
    # ======================================

    @staticmethod
    def current_timestamp():

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ======================================
    # CREATE DATABASE TABLES
    # ======================================

    def create_tables(self):

        try:

            with self.get_connection() as connection:

                cursor = connection.cursor()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS uploads (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        uploaded_at TEXT NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_name TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        registered_at TEXT NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        module_name TEXT NOT NULL,
                        input_text TEXT,
                        result TEXT NOT NULL,
                        confidence REAL,
                        created_at TEXT NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_uploads_uploaded_at
                    ON uploads(uploaded_at)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_faces_person_name
                    ON faces(person_name)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_analyses_created_at
                    ON analyses(created_at)
                """)

            logger.info(
                "Database tables initialized successfully."
            )

        except sqlite3.Error:

            logger.exception(
                "Database table initialization failed."
            )

            raise

    # ======================================
    # ADD UPLOAD
    # ======================================

    def add_upload(self, filename, file_type):

        if not filename or not file_type:

            raise ValueError(
                "Filename and file type are required."
            )

        with self.get_connection() as connection:

            cursor = connection.execute(
                """
                INSERT INTO uploads (
                    filename,
                    file_type,
                    uploaded_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    str(filename),
                    str(file_type),
                    self.current_timestamp()
                )
            )

            upload_id = cursor.lastrowid

        logger.info(
            "Upload record created: %s",
            upload_id
        )

        return upload_id

    # ======================================
    # ADD REGISTERED FACE
    # ======================================

    def add_face(self, person_name, image_path):

        if not person_name or not image_path:

            raise ValueError(
                "Person name and image path are required."
            )

        with self.get_connection() as connection:

            cursor = connection.execute(
                """
                INSERT INTO faces (
                    person_name,
                    image_path,
                    registered_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    str(person_name),
                    str(image_path),
                    self.current_timestamp()
                )
            )

            face_id = cursor.lastrowid

        logger.info(
            "Face record created: %s",
            face_id
        )

        return face_id

    # ======================================
    # GET REGISTERED FACES
    # ======================================

    def get_faces(self):

        with self.get_connection() as connection:

            cursor = connection.execute(
                """
                SELECT
                    id,
                    person_name,
                    image_path,
                    registered_at
                FROM faces
                ORDER BY registered_at DESC
                """
            )

            return [
                dict(row)
                for row in cursor.fetchall()
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

        import json

        if not module_name:

            raise ValueError(
                "Module name is required."
            )

        if result is None:

            raise ValueError(
                "Analysis result cannot be empty."
            )

        serialized_result = json.dumps(
            result,
            ensure_ascii=False,
            default=str
        )

        if confidence is not None:

            confidence = float(confidence)

        with self.get_connection() as connection:

            cursor = connection.execute(
                """
                INSERT INTO analyses (
                    module_name,
                    input_text,
                    result,
                    confidence,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(module_name),
                    str(input_text) if input_text is not None else None,
                    serialized_result,
                    confidence,
                    self.current_timestamp()
                )
            )

            analysis_id = cursor.lastrowid

        logger.info(
            "Analysis record created: %s",
            analysis_id
        )

        return analysis_id

    # ======================================
    # GET ANALYSIS HISTORY
    # ======================================

    def get_analysis_history(self, limit=50):

        import json

        if not isinstance(limit, int) or limit < 1:

            raise ValueError(
                "Limit must be a positive integer."
            )

        limit = min(limit, 500)

        with self.get_connection() as connection:

            cursor = connection.execute(
                """
                SELECT
                    id,
                    module_name,
                    input_text,
                    result,
                    confidence,
                    created_at
                FROM analyses
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            )

            records = cursor.fetchall()

        history = []

        for record in records:

            item = dict(record)

            try:

                item["result"] = json.loads(
                    item["result"]
                )

            except (TypeError, json.JSONDecodeError):

                item["result"] = None

            history.append(item)

        return history

    # ======================================
    # DASHBOARD STATISTICS
    # ======================================

    def get_dashboard_stats(self):

        with self.get_connection() as connection:

            uploads = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM uploads
                """
            ).fetchone()["total"]

            faces = connection.execute(
                """
                SELECT COUNT(DISTINCT person_name) AS total
                FROM faces
                """
            ).fetchone()["total"]

            analyses = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM analyses
                """
            ).fetchone()["total"]

        return {
            "total_uploads": int(uploads),
            "registered_faces": int(faces),
            "total_analyses": int(analyses)
        }


__all__ = [
    "Database"
]
