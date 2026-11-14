
"""
AI Vision Hub — Integration Tests

Tests interactions between the Flask application,
its API routes, and the database layer.
"""

import unittest
from unittest.mock import patch

from app import app, database
from database import Database


class TestApplicationIntegration(unittest.TestCase):
    """Integration tests for the Flask application."""

    @classmethod
    def setUpClass(cls):
        app.config.update(
            TESTING=True,
            SECRET_KEY="integration-test-secret",
        )

    def setUp(self):
        self.client = app.test_client()

    def test_homepage_loads(self):
        """The dashboard should load successfully."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"html", response.data.lower())

    def test_health_endpoint(self):
        """The health endpoint should return a valid response."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        self.assertIsInstance(data, dict)
        self.assertTrue(data.get("success"))
        self.assertIsInstance(data.get("data"), dict)
        self.assertEqual(data["data"].get("status"), "running")

    def test_invalid_api_route(self):
        """Unknown endpoints should return JSON 404 responses."""
        response = self.client.get("/this-route-does-not-exist")

        self.assertEqual(response.status_code, 404)

        data = response.get_json()

        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_sentiment_api_validation(self):
        """Invalid sentiment input should return HTTP 400."""
        response = self.client.post(
            "/sentiment",
            json={"text": ""},
        )

        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertFalse(data["success"])

    def test_ml_prediction_validation(self):
        """Invalid ML input should be rejected."""
        response = self.client.post(
            "/ml-prediction",
            json={"values": []},
        )

        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertFalse(data["success"])

    def test_chatbot_validation(self):
        """An empty chatbot message should be rejected."""
        response = self.client.post(
            "/chatbot",
            json={"message": ""},
        )

        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertFalse(data["success"])

    def test_reset_chat(self):
        """The chat reset endpoint should respond successfully."""
        response = self.client.post("/reset-chat")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])

    def test_analysis_history(self):
        """The analysis history endpoint should return its envelope."""
        response = self.client.get("/analysis-history?limit=5")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        self.assertTrue(data["success"])
        self.assertIn("history", data["data"])
        self.assertIn("count", data["data"])

    def test_wrong_http_method(self):
        """The health endpoint should reject unsupported methods."""
        response = self.client.post("/health")

        self.assertEqual(response.status_code, 405)

        data = response.get_json()
        self.assertFalse(data["success"])


class TestDatabaseIntegration(unittest.TestCase):
    """Integration tests for database operations."""

    def setUp(self):
        self.db = Database(":memory:")

    def test_database_connection(self):
        """The database should create a working connection."""
        with self.db.get_connection() as connection:
            self.assertIsNotNone(connection)

    def test_analysis_save_and_retrieve(self):
        """Saved analysis should be retrievable."""
        self.db.save_analysis(
            module_name="integration_test",
            input_text="Sample input",
            result={"prediction": "positive"},
            confidence=0.95,
        )

        history = self.db.get_analysis_history(limit=10)

        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 1)

        saved_item = next(
            item for item in history
            if item.get("module_name") == "integration_test"
        )

        self.assertEqual(saved_item["input_text"], "Sample input")
        self.assertEqual(
            saved_item["result"]["prediction"],
            "positive",
        )

    def test_upload_recording(self):
        """An uploaded file record should be stored."""
        self.db.add_upload(
            filename="integration-test.jpg",
            file_type="image/jpeg",
        )

        stats = self.db.get_dashboard_stats()

        self.assertGreaterEqual(stats["total_uploads"], 1)

    def test_dashboard_statistics(self):
        """Dashboard statistics should contain expected keys."""
        stats = self.db.get_dashboard_stats()

        self.assertIn("total_uploads", stats)
        self.assertIn("registered_faces", stats)
        self.assertIn("total_analyses", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
