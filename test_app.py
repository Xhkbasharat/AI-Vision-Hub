
import unittest

from unittest.mock import patch

from app import app, database


class TestAIvisionHubApp(unittest.TestCase):

    def setUp(self):

        app.config["TESTING"] = True

        app.config["SECRET_KEY"] = (
            "test-secret-key-for-ai-vision-hub"
        )

        self.client = app.test_client()

    # ======================================
    # DASHBOARD TEST
    # ======================================

    @patch.object(
        database,
        "get_dashboard_stats"
    )
    def test_home_page(self, mock_stats):

        mock_stats.return_value = {
            "total_uploads": 5,
            "registered_faces": 2,
            "total_analyses": 10
        }

        response = self.client.get("/")

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertIn(
            b"AI Vision Hub",
            response.data
        )

    # ======================================
    # HEALTH CHECK TEST
    # ======================================

    def test_health_endpoint(self):

        response = self.client.get(
            "/health"
        )

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.get_json()

        self.assertTrue(
            data["success"]
        )

        self.assertEqual(
            data["result"]["status"],
            "healthy"
        )

    # ======================================
    # UNKNOWN ROUTE TEST
    # ======================================

    def test_unknown_route(self):

        response = self.client.get(
            "/unknown-page"
        )

        self.assertEqual(
            response.status_code,
            404
        )

        data = response.get_json()

        self.assertFalse(
            data["success"]
        )

    # ======================================
    # CHAT RESET TEST
    # ======================================

    def test_reset_chat(self):

        with self.client.session_transaction() as session:

            session["chat_history"] = [
                {
                    "speaker": "user",
                    "message": "Hello"
                }
            ]

        response = self.client.post(
            "/reset-chat"
        )

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.get_json()

        self.assertTrue(
            data["success"]
        )

        with self.client.session_transaction() as session:

            self.assertNotIn(
                "chat_history",
                session
            )

    # ======================================
    # EMPTY SENTIMENT INPUT
    # ======================================

    def test_empty_sentiment_input(self):

        response = self.client.post(
            "/sentiment",
            json={
                "text": ""
            }
        )

        self.assertEqual(
            response.status_code,
            400
        )

    # ======================================
    # INVALID SENTIMENT INPUT TYPE
    # ======================================

    def test_invalid_sentiment_input(self):

        response = self.client.post(
            "/sentiment",
            json={
                "text": 12345
            }
        )

        self.assertEqual(
            response.status_code,
            400
        )

    # ======================================
    # OVERSIZED SENTIMENT INPUT
    # ======================================

    def test_oversized_sentiment_input(self):

        response = self.client.post(
            "/sentiment",
            json={
                "text": "A" * 5001
            }
        )

        self.assertEqual(
            response.status_code,
            400
        )

    # ======================================
    # INVALID ML MODEL
    # ======================================

    def test_invalid_ml_model(self):

        response = self.client.post(
            "/ml-prediction",
            json={
                "values": [
                    5.1,
                    3.5,
                    1.4,
                    0.2
                ],
                "model": "invalid_model"
            }
        )

        self.assertEqual(
            response.status_code,
            400
        )

    # ======================================
    # INVALID HISTORY LIMIT
    # ======================================

    def test_invalid_history_limit(self):

        response = self.client.get(
            "/analysis-history?limit=invalid"
        )

        self.assertEqual(
            response.status_code,
            400
        )

    # ======================================
    # EMPTY CHAT MESSAGE
    # ======================================

    def test_empty_chat_message(self):

        response = self.client.post(
            "/chatbot",
            data={
                "message": ""
            }
        )

        self.assertEqual(
            response.status_code,
            400
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
