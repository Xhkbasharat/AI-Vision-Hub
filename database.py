
import tempfile
import unittest
from pathlib import Path

from database import Database


class TestAIVisionHubDatabase(unittest.TestCase):

    def setUp(self):
        """
        Create an isolated temporary database for each test.
        """

        self.temp_dir = tempfile.TemporaryDirectory()

        self.database_path = (
            Path(self.temp_dir.name) / "test_ai_vision_hub.db"
        )

        self.db = Database()

        # Match the attribute used by database.py.
        self.db.database_path = self.database_path

        # Create tables in the temporary database.
        self.db.create_tables()

    def tearDown(self):
        """
        Clean up the temporary database after each test.
        """

        self.temp_dir.cleanup()

    def test_database_file_created(self):

        self.assertTrue(
            self.database_path.exists()
        )

    def test_add_upload(self):

        upload_id = self.db.add_upload(
            "test_image.jpg",
            "image"
        )

        self.assertIsNotNone(upload_id)
        self.assertGreater(upload_id, 0)

    def test_add_face(self):

        face_id = self.db.add_face(
            "Test Person",
            "data/known_faces/test_person/image_1.jpg"
        )

        self.assertIsNotNone(face_id)
        self.assertGreater(face_id, 0)

        faces = self.db.get_faces()

        self.assertGreaterEqual(
            len(faces),
            1
        )

        self.assertEqual(
            faces[0]["person_name"],
            "Test Person"
        )

    def test_save_analysis(self):

        analysis_id = self.db.save_analysis(
            module_name="object_detection",
            input_text="test_image.jpg",
            result="Detected one object",
            confidence=0.95
        )

        self.assertIsNotNone(analysis_id)
        self.assertGreater(analysis_id, 0)

    def test_analysis_history(self):

        self.db.save_analysis(
            module_name="sentiment_analysis",
            input_text="This is a great project!",
            result="POSITIVE",
            confidence=0.98
        )

        history = self.db.get_analysis_history()

        self.assertGreaterEqual(
            len(history),
            1
        )

        self.assertEqual(
            history[0]["module_name"],
            "sentiment_analysis"
        )

    def test_dashboard_stats(self):

        self.db.add_upload(
            "dashboard_test.jpg",
            "image"
        )

        stats = self.db.get_dashboard_stats()

        self.assertIn("total_uploads", stats)
        self.assertIn("registered_faces", stats)
        self.assertIn("total_analyses", stats)

        self.assertGreaterEqual(
            stats["total_uploads"],
            1
        )

    def test_multiple_uploads(self):

        self.db.add_upload("image1.jpg", "image")
        self.db.add_upload("image2.jpg", "image")

        stats = self.db.get_dashboard_stats()

        self.assertEqual(
            stats["total_uploads"],
            2
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
