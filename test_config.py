
import os
import unittest

from pathlib import Path
from unittest.mock import patch

import config


class TestConfiguration(unittest.TestCase):

    # ======================================
    # APPLICATION SETTINGS
    # ======================================

    def test_application_name(self):

        self.assertEqual(
            config.APP_NAME,
            "AI Vision Hub"
        )

    def test_application_version(self):

        self.assertEqual(
            config.APP_VERSION,
            "1.0.0"
        )

    def test_environment_exists(self):

        self.assertIn(
            config.APP_ENV,
            ["development", "testing", "production"]
        )

    # ======================================
    # SECRET KEY TESTS
    # ======================================

    def test_secret_key_exists(self):

        self.assertTrue(
            config.SECRET_KEY
        )

    def test_secret_key_length(self):

        self.assertGreaterEqual(
            len(config.SECRET_KEY),
            32
        )

    # ======================================
    # UPLOAD CONFIGURATION
    # ======================================

    def test_upload_limit(self):

        self.assertGreater(
            config.MAX_CONTENT_LENGTH,
            0
        )

    def test_allowed_image_extensions(self):

        expected = {
            "jpg",
            "jpeg",
            "png",
            "webp"
        }

        self.assertEqual(
            config.ALLOWED_IMAGE_EXTENSIONS,
            expected
        )

    # ======================================
    # MODEL SETTINGS
    # ======================================

    def test_face_threshold(self):

        self.assertGreater(
            config.FACE_MATCH_THRESHOLD,
            0
        )

    def test_object_threshold(self):

        self.assertGreaterEqual(
            config.OBJECT_CONFIDENCE_THRESHOLD,
            0
        )

        self.assertLessEqual(
            config.OBJECT_CONFIDENCE_THRESHOLD,
            1
        )

    def test_sentiment_model(self):

        self.assertTrue(
            config.SENTIMENT_MODEL_NAME
        )

    def test_chatbot_model(self):

        self.assertTrue(
            config.CHATBOT_MODEL_NAME
        )

    # ======================================
    # DIRECTORY TESTS
    # ======================================

    def test_base_directory_exists(self):

        self.assertTrue(
            config.BASE_DIR.exists()
        )

    def test_data_directory_exists(self):

        self.assertTrue(
            config.DATA_DIR.exists()
        )

    def test_upload_directory_exists(self):

        self.assertTrue(
            config.UPLOAD_FOLDER.exists()
        )

    def test_known_faces_directory_exists(self):

        self.assertTrue(
            config.KNOWN_FACES_FOLDER.exists()
        )

    def test_model_directory_exists(self):

        self.assertTrue(
            config.MODEL_FOLDER.exists()
        )

    def test_database_path_parent(self):

        self.assertEqual(
            config.DATABASE_PATH.parent,
            config.DATA_DIR
        )

    # ======================================
    # PRODUCTION SECURITY TESTS
    # ======================================

    def test_production_debug_is_disabled(self):

        if config.IS_PRODUCTION:

            self.assertFalse(
                config.DEBUG
            )

    def test_production_cookie_security(self):

        if config.IS_PRODUCTION:

            self.assertTrue(
                config.COOKIE_SECURE
            )

    # ======================================
    # CONFIGURATION SUMMARY
    # ======================================

    def test_config_summary(self):

        summary = config.get_config_summary()

        self.assertIsInstance(
            summary,
            dict
        )

        self.assertEqual(
            summary["app_name"],
            config.APP_NAME
        )

        self.assertEqual(
            summary["app_version"],
            config.APP_VERSION
        )

        self.assertEqual(
            summary["environment"],
            config.APP_ENV
        )

        self.assertIn(
            "database_path",
            summary
        )

        self.assertNotIn(
            "secret_key",
            summary
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
