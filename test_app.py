
import unittest
from unittest.mock import patch

import app as application


class TestAIVisionHubApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Configure the Flask application for testing.
        """
        application.app.config.update(
            TESTING=True,
            SECRET_KEY="test-secret-key"
        )

    def setUp(self):
        self.client = application.app.test_client()

    # ======================================
    # HOME PAGE
    # ======================================

    def test_home_page(self):

        with patch.object(
            application.db,
            "get_dashboard_stats",
            return_value={
                "total_uploads": 5,
                "registered_faces": 2,
                "total_analyses": 10
            }
        ):

            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"AI Vision Hub",
            response.data
        )

    # ======================================
    # HEALTH ENDPOINT
    # ======================================

    def test_health_endpoint(self):

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("status"), "healthy")

    # ======================================
    # UNKNOWN ROUTE
    # ======================================

    def test_unknown_route(self):

        response = self.client.get(
            "/this-route-does-not-exist"
        )

        self.assertEqual(response.status_code, 404)

    # ======================================
    # CHAT RESET
    # ======================================

    def test_reset_chat(self):

        with self.client.session_transaction() as session:
            session["chat_history"] = [
                {
                    "speaker": "user",
                    "message": "Hello"
                }
            ]

        response = self.client.post("/reset-chat")

        self.assertEqual(response.status_code, 200)

        with self.client.session_transaction() as session:
            self.assertNotIn("chat_history", session)

    # ======================================
    # OVERSIZED REQUEST
    # ======================================

    def test_oversized_request(self):

        response = self.client.post(
            "/sentiment",
            data={
                "text": "A" * 6000
            },
            content_type="application/json"
        )

        # The route should reject text exceeding
        # its configured 5000-character limit.
        self.assertIn(
            response.status_code,
            [400, 413]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
