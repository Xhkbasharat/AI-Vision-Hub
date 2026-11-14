
import sqlite3
from config import DATABASE_PATH


class Database:

    def __init__(self):
        self.database_path = str(DATABASE_PATH)
        self.create_tables()

    def get_connection(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def create_tables(self):
        connection = self.get_connection()
        cursor = connection.cursor()

        # Uploaded files
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Registered people for face recognition
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # AI analysis results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_name TEXT NOT NULL,
                input_text TEXT,
                result TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()
        connection.close()

    def add_upload(self, filename, file_type):
        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO uploads (filename, file_type)
            VALUES (?, ?)
        """, (filename, file_type))

        connection.commit()
        upload_id = cursor.lastrowid
        connection.close()

        return upload_id

    def add_face(self, person_name, image_path):
        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO faces (person_name, image_path)
            VALUES (?, ?)
        """, (person_name, image_path))

        connection.commit()
        face_id = cursor.lastrowid
        connection.close()

        return face_id

    def get_faces(self):
        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT * FROM faces
            ORDER BY created_at DESC
        """)

        records = [dict(row) for row in cursor.fetchall()]
        connection.close()

        return records

    def save_analysis(
        self,
        module_name,
        input_text,
        result,
        confidence=None
    ):
        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO analysis_results
            (module_name, input_text, result, confidence)
            VALUES (?, ?, ?, ?)
        """, (
            module_name,
            input_text,
            result,
            confidence
        ))

        connection.commit()
        result_id = cursor.lastrowid
        connection.close()

        return result_id

    def get_analysis_history(self, limit=50):
        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT * FROM analysis_results
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        records = [dict(row) for row in cursor.fetchall()]
        connection.close()

        return records

    def get_dashboard_stats(self):
        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM uploads")
        uploads = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM faces")
        faces = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM analysis_results")
        analyses = cursor.fetchone()[0]

        connection.close()

        return {
            "uploads": uploads,
            "registered_faces": faces,
            "analyses": analyses
        }
